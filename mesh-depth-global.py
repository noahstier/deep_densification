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
        f_ll * ((x1 - x) * (y1 - y))
        + f_lr * ((x - x0) * (y1 - y))
        + f_ur * ((x - x0) * (y - y0))
        + f_ul * ((x1 - x) * (y - y0))
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

house_dir = test_house_dirs[1]

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

depthfiles = sorted(glob.glob(os.path.join(depth_imgdir, "*.png")))
depth_imgs = (
    np.stack([cv2.imread(f, cv2.IMREAD_ANYDEPTH) for f in depthfiles], axis=0).astype(
        np.float32
    )
    / 1000
)

model = torch.nn.ModuleDict(
    {
        "query_encoder": torch.nn.Sequential(
            FCLayer(3, 32),
            FCLayer(32, 128),
            FCLayer(128, 256),
            FCLayer(256, 512),
            FCLayer(512, 1024),
        ),
        "mlp": torch.nn.Sequential(
            FCLayer(1280 + 1024, 1024),
            FCLayer(1024, 512),
            FCLayer(512, 256),
            FCLayer(256, 128),
            FCLayer(128, 64),
            FCLayer(64, 32),
            FCLayer(32, 16),
            torch.nn.Linear(16, 1),
        ),
        "cnn": torchvision.models.mobilenet_v2(pretrained=True).features,
    }
).cuda()
model.load_state_dict(torch.load("models/depth-cond-global-87949"))
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

res = 0.02
x = np.arange(-0.2, 0.2, res)
y = np.arange(-0.2, 0.2, res)
z = np.arange(-0.2, 0.2, res)
xx, yy, zz = np.meshgrid(x, y, z)
query_offsets = np.c_[xx.flatten(), yy.flatten(), zz.flatten()]
query_coords = query_offsets / 0.2
query_coords = torch.Tensor(query_coords).cuda()

x = np.arange(len(x), dtype=int)
y = np.arange(len(y), dtype=int)
z = np.arange(len(z), dtype=int)
xx, yy, zz = np.meshgrid(x, y, z)
query_inds = np.c_[xx.flatten(), yy.flatten(), zz.flatten()]

depth_img = depth_imgs[img_ind]
uu, vv = np.meshgrid(
    np.arange(0, depth_img.shape[1], 61), np.arange(0, depth_img.shape[0], 61)
)
anchor_uv = np.c_[uu.flatten(), vv.flatten()]
anchor_xyz_cam = (
    np.linalg.inv(camera_intrinsic) @ np.c_[anchor_uv, np.ones(len(anchor_uv))].T
).T * depth_img[anchor_uv[:, 1], anchor_uv[:, 0], None]
anchor_xyz = (
    camera_extrinsics[img_ind] @ np.c_[anchor_xyz_cam, np.ones(len(anchor_xyz_cam))].T
).T[:, :3]

query_xyz_cam = anchor_xyz_cam[:, None] + query_offsets[None]
query_xyz_cam = torch.Tensor(query_xyz_cam).cuda()

pred_vols = np.empty((len(anchor_uv), *xx.shape))

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

cam2anchor_rot_cuda = torch.Tensor(cam2anchor_rot).cuda()


img_feat = torch.relu(cnn(imgs_t[None, img_ind])).mean([2, 3])

anchors_per_batch = 8
for i in tqdm.trange(int(np.ceil(len(anchor_uv) / anchors_per_batch))):
    start = i * anchors_per_batch
    end = np.minimum(len(pixel_feats), (i + 1) * anchors_per_batch)

    q = model["query_encoder"](
        query_xyz_cam[start:end].reshape(1, (end - start) * len(query_inds), 3)
    )
    f = img_feat[:, None].repeat(1, (end - start) * len(query_inds), 1)

    mlp_input = torch.cat((q, f), dim=-1)

    logits = model["mlp"](mlp_input)

    preds = torch.sigmoid(logits).cpu().numpy().reshape(end - start, len(query_inds))

    pred_vols[start:end, query_inds[:, 0], query_inds[:, 1], query_inds[:, 2]] = preds

meshes = []
for i in range(len(pred_vols)):
    try:
        verts, faces, _, _, = skimage.measure.marching_cubes(pred_vols[i], level=0.5)
    except:
        continue
    faces = np.concatenate((faces, faces[:, ::-1]), axis=0)

    verts_cam_rotated = (verts - np.array(xx.shape) / 2) * res + anchor_xyz_cam_rotated[
        i
    ]
    verts_cam = (cam2anchor_rot[i] @ verts_cam_rotated.T).T
    # verts_cam = (verts - np.array(xx.shape) / 2) * res + anchor_xyz_cam_rotated[i]

    verts_world = (
        camera_extrinsics[img_ind] @ np.c_[verts_cam, np.ones(len(verts_cam))].T
    ).T[:, :3]

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts_world), o3d.utility.Vector3iVector(faces)
    )
    mesh.paint_uniform_color(np.array(plt.cm.jet(np.random.rand()))[:3])
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    meshes.append(mesh)


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

anchor_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(anchor_xyz))
anchor_pcd.paint_uniform_color(np.array([0, 0, 1], dtype=np.float64))

cam_pcd = o3d.geometry.PointCloud(
    o3d.utility.Vector3dVector(camera_extrinsics[img_ind : img_ind + 1, :3, 3])
)
cam_pcd.paint_uniform_color(np.array([1, 0, 0], dtype=np.float64))


mesh = sum(meshes[::5])

geoms = [mesh, gt_pcd, anchor_pcd, cam_pcd]
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
