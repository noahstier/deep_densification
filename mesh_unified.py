import functools
import glob
import importlib
import itertools
import os
import pickle

import cv2
import imageio
import numpy as np
import open3d as o3d
import pickle
import PIL.Image
import scipy.spatial
import scipy.ndimage
import skimage.measure
import torch
import torchvision
import tqdm
import wandb

import fpn
import config
import unet

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

        self.fc = torch.nn.Linear(k_in, k_out, bias=not use_bn)
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


def fc_bn_relu(in_c, out_c):
    return torch.nn.Sequential(
        torch.nn.BatchNorm1d(in_c),
        torch.nn.ReLU(),
        torch.nn.Linear(in_c, out_c, bias=False),
    )


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.coord_encoder = torch.nn.Sequential(
            torch.nn.Linear(3, 32),
            fc_bn_relu(32, 32),
            fc_bn_relu(32, 64),
            fc_bn_relu(64, 128),
        )

        self.offsetter = torch.nn.Sequential(
            fc_bn_relu(256, 256),
            fc_bn_relu(256, 128),
            fc_bn_relu(128, 128),
            fc_bn_relu(128, 128),
            fc_bn_relu(128, 128),
        )

        self.classifier = torch.nn.Sequential(
            fc_bn_relu(128, 128),
            fc_bn_relu(128, 64),
            fc_bn_relu(64, 32),
            fc_bn_relu(32, 16),
            torch.nn.Linear(16, 1, bias=False),
        )

    def forward(self, coords, feats):
        shape = coords.shape[:-1]
        coords = coords.reshape(np.prod([i for i in shape]), coords.shape[-1])
        feats = feats.reshape(np.prod([i for i in shape]), feats.shape[-1])

        encoded_coords = self.coord_encoder(coords)
        offset = self.offsetter(torch.cat((encoded_coords, feats), dim=-1)) + feats
        logits = self.classifier(offset)
        logits = logits.reshape(*shape, -1)
        return logits


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

included_classes = [5, 11, 20, 67, 72]
# included_classes = [67]
"""
with open("per_img_classes.pkl", "rb") as f:
    per_img_classes = pickle.load(f)
tups = []
a = {h: per_img_classes[os.path.basename(h)] for h in test_house_dirs}
for house_dir, b in a.items():
    for img_name, d in b.items():
        n = np.sum([c in d for c in included_classes])
        if n >= 3:
            tups.append((os.path.basename(house_dir), img_name, n))

tups = sorted(tups, key=lambda tup: tup[-1], reverse=True)
"""

house_dir = test_house_dirs[-2]
img_ind = 37

rgb_imgdir = os.path.join(house_dir, "imgs/color")
depth_imgdir = os.path.join(house_dir, "imgs/depth")
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


cat_imgs = np.stack(
    [
        PIL.Image.open(
            os.path.join(cat_imgdir, ims[im_id].name.replace("jpg", "png"))
        ).transpose(PIL.Image.FLIP_TOP_BOTTOM)
        for im_id in im_ids
    ],
    axis=0,
)

depth_imgs = (
    np.stack(
        [
            cv2.imread(
                os.path.join(depth_imgdir, ims[im_id].name.replace("jpg", "png")),
                cv2.IMREAD_ANYDEPTH,
            )
            for im_id in im_ids
        ],
        axis=0,
    ).astype(np.float32)
    / 1000
)


input_height = imgs_t.shape[2]
input_width = imgs_t.shape[3]

model = torch.nn.ModuleDict(
    {
        "cnn": fpn.FPN(input_height, input_width, 1),
        "mlp": MLP(),
        # "query_encoder": torch.nn.Sequential(
        #     FCLayer(3, 64), FCLayer(64), FCLayer(64), FCLayer(64, 128),
        # ),
        # "mlp": torch.nn.Sequential(
        #    FCLayer(67, 128),
        #    FCLayer(128, 256),
        #    FCLayer(256, 512),
        #    FCLayer(512),
        #    FCLayer(512),
        #    FCLayer(512, 256),
        #    FCLayer(256, 64),
        #    FCLayer(64, 16),
        #    torch.nn.Linear(16, 1),
        # ),
        # "mlp": torch.nn.Sequential(
        #     FCLayer(256),
        #     FCLayer(256, 512),
        #     FCLayer(512, 1024),
        #     FCLayer(1024),
        #     FCLayer(1024),
        #     FCLayer(1024, 512),
        #     FCLayer(512, 256),
        #     FCLayer(256, 64),
        #     FCLayer(64, 16),
        #     torch.nn.Linear(16, 1),
        # ),
        # "cnn": torchvision.models.mobilenet_v2(pretrained=True).features[:7],
        # "cnn": unet.UNet(n_channels=3),
        "cnn": fpn.FPN(input_height, input_width, 1),
    }
)
# model.load_state_dict(torch.load("models/sofa-only")['model'])
model.load_state_dict(torch.load("models/depth-cond-anchor-212834")["model"])
model = model.cuda()
model.eval()
model.requires_grad_(False)

cnn = model["cnn"]

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

depth_img = depth_imgs[img_ind]
uu, vv = np.meshgrid(
    np.arange(50, depth_img.shape[1], 40), np.arange(50, depth_img.shape[0], 40)
)

im = ims[img_ind]
visible_inds = (im.point3D_ids != -1) & (np.array([i in pts for i in im.point3D_ids]))
visible_pt_ids = im.point3D_ids[visible_inds]
visible_pt_uv = im.xys[visible_inds]

if False:
    # depth pixels, all classes
    anchor_inds = np.c_[uu.flatten(), vv.flatten()]
    anchor_uv = anchor_inds + .5
elif False:
    # depth pixels, included classes
    included_mask = np.sum([(cat_imgs[img_ind] == c) for c in included_classes], axis=0)
    anchor_inds = np.argwhere(included_mask > 0)[:, [1, 0]]
    anchor_inds = anchor_inds[
        np.random.choice(np.arange(len(anchor_inds)), size=200, replace=False)
    ]
    anchor_uv = anchor_inds + .5
elif True:
    # SFM points, all classes
    anchor_uv = visible_pt_uv
    anchor_inds = np.floor(anchor_uv).astype(int)

anchor_xyz_cam = (
    np.linalg.inv(camera_intrinsic) @ np.c_[anchor_uv, np.ones(len(anchor_uv))].T
).T * depth_img[anchor_inds[:, 1], anchor_inds[:, 0], None]
anchor_xyz = (
    camera_extrinsics[img_ind] @ np.c_[anchor_xyz_cam, np.ones(len(anchor_xyz_cam))].T
).T[:, :3]

anchor_xyz_cam_u = anchor_xyz_cam / np.linalg.norm(
    anchor_xyz_cam, axis=-1, keepdims=True
)
center_pixel_u = np.array([0, 0, 1])
cross = np.cross(center_pixel_u, anchor_xyz_cam_u)
cross /= np.linalg.norm(cross, axis=-1, keepdims=True)
dot = np.dot(center_pixel_u, anchor_xyz_cam_u.T)
axis = cross
angle = np.arccos(dot)
cam2anchor_rot = scipy.spatial.transform.Rotation.from_rotvec(
    axis * angle[:, None]
).as_matrix()

anchor_xyz_cam_rotated = np.stack(
    [
        (np.linalg.inv(cam2anchor_rot[i]) @ anchor_xyz_cam[i]).T
        for i in range(len(anchor_xyz_cam))
    ],
    axis=0,
)

maxbounds = np.max(anchor_xyz, axis=0) + 0.2
minbounds = np.min(anchor_xyz, axis=0) - 0.2

res = 0.02
x = np.arange(minbounds[0], maxbounds[0], res)
y = np.arange(minbounds[1], maxbounds[1], res)
z = np.arange(minbounds[2], maxbounds[2], res)
xx, yy, zz = np.meshgrid(x, y, z)
query_xyz = np.c_[xx.flatten(), yy.flatten(), zz.flatten()]

x = np.arange(len(x), dtype=int)
y = np.arange(len(y), dtype=int)
z = np.arange(len(z), dtype=int)
xx, yy, zz = np.meshgrid(x, y, z)
query_inds = np.c_[xx.flatten(), yy.flatten(), zz.flatten()]

pred_vols = np.ones((len(anchor_uv), *xx.shape)) * np.nan


img_feat, _ = cnn(imgs_t[None, img_ind])

anchor_uv_t = (
    anchor_uv
    / [depth_img.shape[1], depth_img.shape[0]]
    * [img_feat.shape[3], img_feat.shape[2]]
)
pixel_feats = interp_img(img_feat[0], torch.Tensor(anchor_uv_t).cuda()).T


for i in tqdm.trange(len(anchor_uv)):
    in_range_inds = np.linalg.norm((anchor_xyz[i] - query_xyz), axis=-1) < 0.2
    cur_query_xyz = query_xyz[in_range_inds]
    cur_query_inds = query_inds[in_range_inds]

    query_xyz_cam = (
        np.linalg.inv(camera_extrinsics[img_ind])
        @ np.c_[cur_query_xyz, np.ones(len(cur_query_xyz))].T
    ).T[:, :3]
    query_xyz_rotated = (np.linalg.inv(cam2anchor_rot[i]) @ query_xyz_cam.T).T
    anchor_xyz_rotated = np.linalg.inv(cam2anchor_rot[i]) @ anchor_xyz_cam[i]
    query_coords = (query_xyz_rotated - anchor_xyz_rotated) / 0.2
    query_coords = torch.Tensor(query_coords).cuda()

    logits = model["mlp"](
        query_coords[None],
        pixel_feats[None, None, i].repeat(1, query_coords.shape[0], 1),
    )

    preds = torch.sigmoid(logits).cpu().numpy()[0, ..., 0]

    pred_vols[
        i, cur_query_inds[:, 1], cur_query_inds[:, 0], cur_query_inds[:, 2]
    ] = preds


pred_vol = np.nanmean(pred_vols, axis=0)
mask = ~np.isnan(pred_vol)
verts, faces, _, _, = skimage.measure.marching_cubes(
    pred_vol,
    level=0.5,
    mask=scipy.ndimage.morphology.binary_erosion(mask, iterations=2),
)
verts = verts - np.array(xx.shape) / 2
verts = verts[:, [1, 0, 2]]
verts = verts * res + (maxbounds + minbounds) / 2
faces = np.concatenate((faces, faces[:, ::-1]), axis=0)

pred_mesh = o3d.geometry.TriangleMesh(
    o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces)
)
pred_mesh.compute_vertex_normals()
pred_mesh.compute_triangle_normals()

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

gt_pcd_cam = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt_xyz_cam[inds]))
gt_pcd_cam.colors = o3d.utility.Vector3dVector(gt_rgb)

cam_pcd = o3d.geometry.PointCloud(
    o3d.utility.Vector3dVector(camera_extrinsics[img_ind : img_ind + 1, :3, 3])
)
cam_pcd.paint_uniform_color(np.array([1, 0, 0], dtype=np.float64))

anchor_spheres = sum(
    [
        o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        .translate(anchor_xyz[i])
        .paint_uniform_color(np.array([0, 0, 1]))
        .compute_vertex_normals()
        for i in range(len(anchor_xyz))
    ]
)

geoms = [pred_mesh, gt_pcd, anchor_spheres, cam_pcd]
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


"""

pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector((query_inds - np.array(xx.shape) / 2) * res + anchor_xyz_cam_rotated[0]))
pcd.colors = o3d.utility.Vector3dVector(plt.cm.jet(pred_vols[0].flatten())[:, :3])
o3d.visualization.draw_geometries([mesh, pcd])
"""
