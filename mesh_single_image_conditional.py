import functools
import glob
import importlib
import itertools
import os
import pickle

import imageio
import numpy as np
import open3d as o3d
import PIL.Image
import scipy.spatial
import scipy.ndimage
import skimage.measure
import torch
import torchvision
import tqdm
import unet
import wandb

import config

torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(1)


spec = importlib.util.spec_from_file_location(
    "colmap_reader", config.colmap_reader_script
)
colmap_reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(colmap_reader)


class FCLayer(torch.nn.Module):
    def __init__(self, k_in, k_out=None, use_bn=True):
        super(FCLayer, self).__init__()
        if k_out is None:
            k_out = k_in
        if k_out == k_in:
            self.residual = True
        else:
            self.residual = False

        self.use_bn = use_bn

        self.fc = torch.nn.Linear(k_in, k_out)
        if self.use_bn:
            self.bn = torch.nn.BatchNorm1d(k_out)

    def forward(self, inputs):
        x = inputs
        shape = x.shape[:-1]

        x = x.reshape(np.prod([i for i in shape]), x.shape[-1])
        x = self.fc(x)
        if self.use_bn:
            x = self.bn(x)
        x = torch.relu(x)
        x = x.reshape(*shape, x.shape[-1])
        if self.residual:
            x = x + inputs
        return x


def interp_img(img, xy):
    x = xy[:, 0]
    y = xy[:, 1]

    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()

    x1 = torch.clamp_max(x0 + 1, img.shape[2] - 1)
    y1 = torch.clamp_max(y0 + 1, img.shape[1] - 1)

    assert torch.all(
        (y0 >= 0) & (y0 <= img.shape[1] - 1) & (x0 >= 0) & (x0 <= img.shape[2] - 1)
    )

    f_ll = img[:, y0, x0]
    f_lr = img[:, y0, x1]
    f_ul = img[:, y1, x0]
    f_ur = img[:, y1, x1]

    interped = (
        f_ll * ((x - x0) * (y - y0))
        + f_lr * ((x1 - x) * (y - y0))
        + f_ur * ((x1 - x) * (y1 - y))
        + f_ul * ((x - x0) * (y1 - y))
    )
    return interped


def positional_encoding(xyz, L):
    encoding = []
    for l in range(L):
        encoding.append(np.sin(2 ** l ** np.pi * xyz))
        encoding.append(np.cos(2 ** l ** np.pi * xyz))
    encoding = np.concatenate(encoding, axis=-1)
    return encoding


house_dirs = sorted(
    [
        d
        for d in glob.glob(os.path.join(config.dset_dir, "*"))
        if os.path.exists(os.path.join(d, "fusion.npz"))
        and os.path.exists(os.path.join(d, "poses.npz"))
    ]
)
test_house_dirs = house_dirs[:10]
train_house_dirs = house_dirs[10:]

house_dir = test_house_dirs[3]

rgb_imgdir = os.path.join(house_dir, "imgs/color")
cat_imgdir = os.path.join(house_dir, "imgs/category")
fusion_npz = np.load(os.path.join(house_dir, "fusion.npz"))
sfm_imfile = os.path.join(house_dir, "sfm/sparse/auto/images.bin")
sfm_ptfile = os.path.join(house_dir, "sfm/sparse/auto/points3D.bin")

ims = colmap_reader.read_images_binary(sfm_imfile)
pts = colmap_reader.read_points3d_binary(sfm_ptfile)

pts = {
    pt_id: pt for pt_id, pt in pts.items() if pt.error < 1 and len(pt.image_ids) >= 5
}

im_ids = sorted(ims.keys())
pt_ids = sorted(pts.keys())

sfm_xyz = np.stack([pts[i].xyz for i in pt_ids], axis=0)

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

pil_imgs = [PIL.Image.open(os.path.join(rgb_imgdir, ims[i].name)) for i in im_ids]
imgs = np.stack([np.asarray(img) for img in pil_imgs], axis=0)
imgs_t = torch.stack([transform(img) for img in pil_imgs], dim=0).cuda()
imheight, imwidth, _ = imgs[0].shape

model = torch.nn.ModuleDict(
    {
        "mlp": torch.nn.Sequential(
            FCLayer(1286, 1024),
            FCLayer(1024, 512),
            FCLayer(512),
            FCLayer(512),
            FCLayer(512),
            FCLayer(512),
            FCLayer(512, 256),
            FCLayer(256, 64),
            FCLayer(64, 16),
            torch.nn.Linear(16, 1),
            # torch.nn.Linear(16, 1),
            # FCLayer(1340, 256),
            # FCLayer(256),
            # FCLayer(256),
            # FCLayer(256),
            # FCLayer(256),
            # FCLayer(256),
            # FCLayer(256),
            # FCLayer(256, 64),
            # FCLayer(64, 16),
            # torch.nn.Linear(16, 1),
        ),
        # 'cnn': torchvision.models.mobilenet_v2(pretrained=True).features[:7],
        "cnn": torchvision.models.mobilenet_v2(pretrained=True).features,
    }
)
model.load_state_dict(torch.load("models/im-cond"))
model = model.cuda()
model.eval()
model.requires_grad_(False)

cnn = model["cnn"]
featheight, featwidth = cnn(imgs_t[:1]).shape[2:]

intr_file = os.path.join(house_dir, "camera_intrinsics")
camera_intrinsic = np.loadtxt(intr_file)[0].reshape(3, 3)
camera_extrinsics = np.zeros((len(imgs), 4, 4))
for i, im_id in enumerate(im_ids):
    im = ims[im_id]
    qw, qx, qy, qz = im.qvec
    r = scipy.spatial.transform.Rotation.from_quat([qx, qy, qz, qw]).as_matrix().T
    t = np.linalg.inv(-r.T) @ im.tvec
    camera_extrinsics[i] = np.array(
        [[*r[0], t[0]], [*r[1], t[1]], [*r[2], t[2]], [0, 0, 0, 1]]
    )

img_ind = 10

reproj_xyz = fusion_npz["reprojection_samples"]
maxbounds = np.percentile(reproj_xyz, 98, axis=0) + 0.1
minbounds = np.percentile(reproj_xyz, 2, axis=0) + 0.1

res = 0.02
x = np.arange(minbounds[0], maxbounds[0], res)
y = np.arange(minbounds[1], maxbounds[1], res)
z = np.arange(minbounds[2], maxbounds[2], res)
xx, yy, zz = np.meshgrid(x, y, z)
query_pts = np.c_[xx.flatten(), yy.flatten(), zz.flatten()]

x = np.arange(len(x), dtype=int)
y = np.arange(len(y), dtype=int)
z = np.arange(len(z), dtype=int)
xx, yy, zz = np.meshgrid(x, y, z)
query_inds = np.c_[xx.flatten(), yy.flatten(), zz.flatten()]

query_pts_cam = (
    np.linalg.inv(camera_extrinsics[img_ind])
    @ np.c_[query_pts, np.ones(len(query_pts))].T
).T[:, :3]

query_uv = (camera_intrinsic @ query_pts_cam.T).T
query_uv = query_uv[:, :2] / query_uv[:, 2:]

inds = (
    (query_pts_cam[:, 2] > 0)
    & (query_uv[:, 0] >= 0)
    & (query_uv[:, 1] >= 0)
    & (query_uv[:, 0] < imwidth)
    & (query_uv[:, 1] < imheight)
)
query_pts = query_pts[inds]
query_pts_cam = query_pts_cam[inds]
query_uv = query_uv[inds]
query_inds = query_inds[inds]

query_uv_t = query_uv / [imwidth, imheight] * [featwidth, featheight]

normed_query_pts_cam = query_pts_cam - np.array([-3.47185991, -2.54137404, 0.022674])
normed_query_pts_cam /= np.array([6.63462202, 3.68160095, 6.79344803])
normed_query_pts_cam = normed_query_pts_cam * 2 - 1
encoded_query_pts = positional_encoding(normed_query_pts_cam, L=1)

encoded_query_pts = torch.Tensor(encoded_query_pts).cuda()
pred_vol = np.zeros(xx.shape, dtype=np.float32)
pred_mask = np.zeros(xx.shape, dtype=np.bool)

img_feat = torch.relu(cnn(imgs_t[None, img_ind]))
img_feat = torch.mean(img_feat, dim=2)
img_feat = torch.mean(img_feat, dim=2)

for i in tqdm.trange(int(np.ceil(len(query_pts) / config.query_pts_per_batch))):
    start = i * config.query_pts_per_batch
    end = (i + 1) * config.query_pts_per_batch

    # pixel_feats = interp_img(img_feat[0], torch.Tensor(query_uv_t[start:end]).cuda()).T
    # mlp_input = torch.cat((encoded_query_pts[start:end], pixel_feats), dim=1)
    # logits = model['mlp'](mlp_input)

    # query_xyz_world = query_pts[start:end].copy()
    # query_xyz_world -= np.array([3.75210289e+01, 4.46640313e-02, 4.68986430e+01])
    # query_xyz_world /= np.array([7.52712135, 2.74440703, 6.09322482])
    # query_xyz_world = query_xyz_world * 2 - 1
    # query_xyz_world = torch.Tensor(query_xyz_world).cuda()
    query_coords = encoded_query_pts[start:end]

    img_ind_encoding = img_ind / len(imgs) * 2 - 1
    img_ind_encoding = positional_encoding(np.array([img_ind_encoding]), L=2)
    img_ind_encoding = np.tile(img_ind_encoding[None], (len(query_coords), 1))
    img_ind_encoding = torch.Tensor(img_ind_encoding).cuda()

    mlp_input = torch.cat((query_coords, img_feat.repeat(len(query_coords), 1)), dim=1)

    logits = model["mlp"](mlp_input)

    preds = torch.sigmoid(logits).cpu().numpy()[..., 0]
    qi = query_inds[start:end]
    pred_vol[qi[:, 1], qi[:, 0], qi[:, 2]] = preds
    pred_mask[qi[:, 1], qi[:, 0], qi[:, 2]] = 1

verts, faces, _, _, = skimage.measure.marching_cubes(
    pred_vol,
    level=0.5,
    mask=scipy.ndimage.morphology.binary_erosion(pred_mask, iterations=3),
)
verts = verts[:, [1, 0, 2]]
faces = np.concatenate((faces, faces[:, ::-1]), axis=0)

verts = verts * res + minbounds

mesh = o3d.geometry.TriangleMesh(
    o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces)
)
mesh.compute_vertex_normals()
mesh.compute_triangle_normals()

gt_xyz = fusion_npz["tsdf_point_cloud"][:, :3]
gt_rgb = fusion_npz["tsdf_point_cloud"][:, 3:] / 255
gt_xyz_cam = (
    np.linalg.inv(camera_extrinsics[img_ind]) @ np.c_[gt_xyz, np.ones(len(gt_xyz))].T
).T[:, :3]
gt_uv = (camera_intrinsic @ gt_xyz_cam.T).T
gt_uv = gt_uv[:, :2] / gt_uv[:, 2:]
inds = (
    (gt_xyz_cam[:, 2] > 0)
    & (gt_uv[:, 0] >= 0)
    & (gt_uv[:, 0] <= imwidth)
    & (gt_uv[:, 1] >= 0)
    & (gt_uv[:, 1] <= imheight)
)
gt_xyz = gt_xyz[inds]
gt_rgb = gt_rgb[inds]
gt_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt_xyz))
gt_pcd.colors = o3d.utility.Vector3dVector(gt_rgb)

query_pts = fusion_npz["query_pts"]
query_tsdf = fusion_npz["query_tsdf"]

query_pts_cam = (
    np.linalg.inv(camera_extrinsics[img_ind])
    @ np.c_[query_pts, np.ones(len(query_pts))].T
).T[:, :3]
query_uv = (camera_intrinsic @ query_pts_cam.T).T
query_uv = query_uv[:, :2] / query_uv[:, 2:]
inds = (
    (query_pts_cam[:, 2] > 0)
    & (query_uv[:, 0] >= 0)
    & (query_uv[:, 0] <= imwidth)
    & (query_uv[:, 1] >= 0)
    & (query_uv[:, 1] <= imheight)
)
query_pts = query_pts[inds]
query_tsdf = query_tsdf[inds]
query_rgb = np.array([[1, 0, 0], [0, 0, 1]])[(query_tsdf > 0).astype(int)]
query_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(query_pts))
query_pcd.colors = o3d.utility.Vector3dVector(query_rgb)

geoms = [mesh, gt_pcd, query_pcd]
visibility = [True] * len(geoms)


def toggle_geom(vis, geom_ind):
    if visibility[geom_ind]:
        vis.remove_geometry(geoms[geom_ind], reset_bounding_box=False)
        visibility[geom_ind] = False
    else:
        vis.add_geometry(geoms[geom_ind], reset_bounding_box=False)
        visibility[geom_ind] = True


callbacks = {}
for i in range(len(geoms)):
    callbacks[ord(str(i + 1))] = functools.partial(toggle_geom, geom_ind=i)
o3d.visualization.draw_geometries_with_key_callbacks(geoms, callbacks)
