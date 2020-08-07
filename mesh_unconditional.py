import functools
import glob
import importlib
import itertools
import os

import imageio
import numpy as np
import open3d as o3d
import PIL.Image
import scipy.spatial
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


def interp_maps(maps, xy):
    x = xy[:, 0]
    y = xy[:, 1]

    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()

    x1 = torch.clamp_max(x0 + 1, maps.shape[3] - 1)
    y1 = torch.clamp_max(y0 + 1, maps.shape[2] - 1)

    batch_inds = torch.arange(maps.shape[0])

    assert torch.all(
        (y0 >= 0) & (y0 <= maps.shape[2] - 1) & (x0 >= 0) & (x0 <= maps.shape[3] - 1)
    )

    f_ll = maps[batch_inds, :, y0, x0]
    f_lr = maps[batch_inds, :, y0, x1]
    f_ul = maps[batch_inds, :, y1, x0]
    f_ur = maps[batch_inds, :, y1, x1]

    interped = (
        f_ll * ((x - x0) * (y - y0))[:, None]
        + f_lr * ((x1 - x) * (y - y0))[:, None]
        + f_ur * ((x1 - x) * (y1 - y))[:, None]
        + f_ul * ((x - x0) * (y1 - y))[:, None]
    )
    return interped


def positional_encoding(xyz, L):
    encoding = []
    for l in range(L):
        encoding.append(np.sin(2 ** l ** np.pi * xyz))
        encoding.append(np.cos(2 ** l ** np.pi * xyz))
    encoding = np.concatenate(encoding, axis=-1)
    return encoding


rgb_imgdir = os.path.join(config.house_dir, "imgs/color")
cat_imgdir = os.path.join(config.house_dir, "imgs/category")
fusion_npz = np.load(os.path.join(config.house_dir, "fusion.npz"))
sfm_imfile = os.path.join(config.house_dir, "sfm/sparse/auto/images.bin")
sfm_ptfile = os.path.join(config.house_dir, "sfm/sparse/auto/points3D.bin")


model = torch.load("models/pos1")
model.eval()
model.requires_grad_(False)

minbounds = np.array([43, 0, 52])
maxbounds = np.array([45, 2, 53])
# minbounds = np.array([3.75210289e+01, 4.46640313e-02, 4.68986430e+01])
# maxbounds = np.array([45.04815025,  2.78907107, 52.99186782])

res = 0.02
x = np.arange(minbounds[0], maxbounds[0], res)
y = np.arange(minbounds[1], maxbounds[1], res)
z = np.arange(minbounds[2], maxbounds[2], res)
xx, yy, zz = np.meshgrid(x, y, z)
query_pts = np.c_[xx.flatten(), yy.flatten(), zz.flatten()]

query_pts -= np.array([3.75212101e01, 4.49167082e-02, 4.68986196e01])
query_pts /= np.array([7.52686061, 2.73002229, 6.0932354])
query_pts = query_pts * 2 - 1

query_pts = positional_encoding(query_pts, L=1)

query_pts = torch.Tensor(query_pts).cuda()

x = np.arange(len(x), dtype=int)
y = np.arange(len(y), dtype=int)
z = np.arange(len(z), dtype=int)
xx, yy, zz = np.meshgrid(x, y, z)
query_inds = np.c_[xx.flatten(), yy.flatten(), zz.flatten()]

pred_vol = np.zeros(xx.shape, dtype=np.float32)

for i in tqdm.trange(int(np.ceil(len(query_pts) / config.query_pts_per_batch))):
    start = i * config.query_pts_per_batch
    end = (i + 1) * config.query_pts_per_batch
    query_xyz = query_pts[start:end]
    preds = torch.sigmoid(model(query_xyz)).cpu().numpy()[..., 0]
    qi = query_inds[start:end]
    pred_vol[qi[:, 1], qi[:, 0], qi[:, 2]] = preds

verts, faces, _, _, = skimage.measure.marching_cubes(pred_vol, level=0.5)
faces = np.concatenate((faces, faces[:, ::-1]), axis=0)

mesh = o3d.geometry.TriangleMesh(
    o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces)
)
mesh.compute_vertex_normals()
mesh.compute_triangle_normals()

gt_xyz = fusion_npz["tsdf_point_cloud"][:, :3]
gt_rgb = fusion_npz["tsdf_point_cloud"][:, 3:] / 255
inds = (
    (gt_xyz[:, 0] > minbounds[0])
    & (gt_xyz[:, 0] < maxbounds[0])
    & (gt_xyz[:, 1] > minbounds[1])
    & (gt_xyz[:, 1] < maxbounds[1])
    & (gt_xyz[:, 2] > minbounds[2])
    & (gt_xyz[:, 2] < maxbounds[2])
)
gt_xyz = gt_xyz[inds]
gt_rgb = gt_rgb[inds]
gt_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt_xyz))
gt_pcd.colors = o3d.utility.Vector3dVector(gt_rgb)

query_pts = fusion_npz["query_pts"]
query_tsdf = fusion_npz["query_tsdf"]
inds = (
    (query_pts[:, 0] > minbounds[0])
    & (query_pts[:, 0] < maxbounds[0])
    & (query_pts[:, 1] > minbounds[1])
    & (query_pts[:, 1] < maxbounds[1])
    & (query_pts[:, 2] > minbounds[2])
    & (query_pts[:, 2] < maxbounds[2])
)
query_pts = query_pts[inds]
query_tsdf = query_tsdf[inds]
query_rgb = np.array([[1, 0, 0], [0, 0, 1]])[(query_tsdf > 0).astype(int)]
query_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(query_pts))
query_pcd.colors = o3d.utility.Vector3dVector(query_rgb)


o3d.visualization.draw_geometries([query_pcd, gt_pcd])
o3d.visualization.draw_geometries([mesh])
