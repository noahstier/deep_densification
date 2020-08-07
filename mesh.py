import functools
import glob
import importlib
import itertools
import os

import imageio
import numpy as np
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


mlp = torch.nn.Sequential(
    FCLayer(99, 64), FCLayer(64), FCLayer(64), FCLayer(64), FCLayer(64), FCLayer(64),
)
occ_classifier = torch.nn.Sequential(
    FCLayer(64, 64), FCLayer(64, 32), FCLayer(32, 8), torch.nn.Linear(8, 1)
)
pointnet_mlp = torch.nn.Sequential(
    FCLayer(3, 64), FCLayer(64), FCLayer(64), FCLayer(64), FCLayer(64), FCLayer(64),
)

rgb_imgdir = os.path.join(config.house_dir, "imgs/color")
cat_imgdir = os.path.join(config.house_dir, "imgs/category")
fusion_npz = np.load(os.path.join(config.house_dir, "fusion.npz"))
sfm_imfile = os.path.join(config.house_dir, "sfm/sparse/auto/images.bin")
sfm_ptfile = os.path.join(config.house_dir, "sfm/sparse/auto/points3D.bin")

query_pts = fusion_npz["query_pts"]
query_tsdf = fusion_npz["query_tsdf"]
reprojection_samples = fusion_npz["reprojection_samples"]

ims = colmap_reader.read_images_binary(sfm_imfile)
pts = colmap_reader.read_points3d_binary(sfm_ptfile)

pts = {
    pt_id: pt for pt_id, pt in pts.items() if pt.error < 1 and len(pt.image_ids) >= 5
}

im_ids = sorted(ims.keys())
pt_ids = sorted(pts.keys())

sfm_xyz = np.stack([pts[i].xyz for i in pt_ids], axis=0)
kdtree = scipy.spatial.KDTree(sfm_xyz)

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

"""
cat_imgfiles = sorted(glob.glob(os.path.join(cat_imgdir, "*.pfm")))
cat_transform = torchvision.transforms.Resize(224, PIL.Image.NEAREST)
cat_imgs = np.stack(
    [cat_transform(PIL.Image.fromarray(imageio.imread(f)[::-1])) for f in cat_imgfiles],
    axis=0,
).astype(np.uint8)
cats = sorted(np.unique(cat_imgs).astype(np.uint8))
a = np.zeros(np.max(cats) + 1, dtype=np.uint8)
a[cats] = np.arange(len(cats))
cat_gt = a[cat_imgs]
cat_gt = torch.Tensor(cat_gt).long().cuda()
"""

intr_file = os.path.join(config.house_dir, "camera_intrinsics")
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
query_neighborhood_radius = config.query_neighborhood_radius
query_pts_per_batch = config.query_pts_per_batch
imgs_per_anchor_pt = config.imgs_per_anchor_pt
anchor_pts_per_query = config.anchor_pts_per_query
n_pointnet_pts = config.n_pointnet_pts

res = 0.05
minbounds = np.min(sfm_xyz, axis=0)
maxbounds = np.max(sfm_xyz, axis=0)
minbounds = np.array([44, 0, 47])
maxbounds = np.array([45, 2, 50])
x = np.arange(minbounds[0], maxbounds[0], res)
y = np.arange(minbounds[1], maxbounds[1], res)
z = np.arange(minbounds[2], maxbounds[2], res)
xx, yy, zz = np.meshgrid(x, y, z)
query_coords = np.c_[xx.flatten(), yy.flatten(), zz.flatten()]
query_inds = np.argwhere(xx)

pred_volume = np.zeros(xx.shape, dtype=np.float32)
pred_mask = np.zeros(xx.shape, dtype=np.bool)

print("filtering query coords")
sfm_xyz = torch.Tensor(sfm_xyz).cuda()
keep = np.zeros(len(query_coords), dtype=bool)
for i in tqdm.trange(int(np.ceil(len(query_coords) / 10_000))):
    start = i * 10_000
    end = (i + 1) * 10_000
    q = torch.Tensor(query_coords[start:end]).cuda()
    dists = torch.norm(q[:, None] - sfm_xyz[None], dim=-1)
    keep[start:end] = (
        (torch.sum(dists < query_neighborhood_radius, dim=1) >= anchor_pts_per_query)
        .cpu()
        .numpy()
    )

sfm_xyz = sfm_xyz.cpu().numpy()
query_coords = query_coords[keep]
query_inds = query_inds[keep]

print("pre-querying nns")
_, _anchor_pt_inds = kdtree.query(query_coords, k=anchor_pts_per_query, eps=0.1)
_, _local_sfm_inds = kdtree.query(sfm_xyz, k=n_pointnet_pts, eps=0.1)

pred_mask[query_inds[:, 0], query_inds[:, 1], query_inds[:, 2]] = True


model = torch.load("model")

model.eval()
model.requires_grad_(False)

print("pre-extracting image features")
img_feats = []
for img_t in tqdm.tqdm(imgs_t):
    img_feats.append(model["cnn"](img_t[None]).cpu())
img_feats = torch.cat(img_feats, dim=0)

feat_height, feat_width = model["cnn"](imgs_t[:1]).shape[2:]

indices = np.arange(len(query_coords), dtype=int)


for batch_ind in tqdm.trange(len(query_coords) // query_pts_per_batch):
    start = batch_ind * query_pts_per_batch
    end = (batch_ind + 1) * query_pts_per_batch

    cur_query_coords = query_coords[start:end]
    cur_query_inds = query_inds[start:end]

    anchor_pt_inds = _anchor_pt_inds[start:end]

    anchor_pt_ids = np.array(pt_ids)[anchor_pt_inds]
    anchor_xyz = sfm_xyz[anchor_pt_inds]

    local_sfm_inds = _local_sfm_inds[anchor_pt_inds]
    local_sfm_xyz = sfm_xyz[local_sfm_inds]

    anchor_img_inds = np.zeros((*anchor_pt_ids.shape, imgs_per_anchor_pt), dtype=int)
    anchor_uv = np.zeros((*anchor_pt_ids.shape, imgs_per_anchor_pt, 2))
    for i in range(anchor_pt_ids.shape[0]):
        for j in range(anchor_pt_ids.shape[1]):
            anchor_pt = pts[anchor_pt_ids[i, j]]
            inds = np.random.choice(
                np.arange(len(anchor_pt.image_ids)),
                size=imgs_per_anchor_pt,
                replace=False,
            )
            anchor_img_ids = anchor_pt.image_ids[inds]
            point2D_idxs = anchor_pt.point2D_idxs[inds]
            anchor_uv[i, j] = [
                ims[img_id].xys[point2D_idx]
                for img_id, point2D_idx in zip(anchor_img_ids, point2D_idxs)
            ]
            for k in range(imgs_per_anchor_pt):
                anchor_img_inds[i, j, k] = im_ids.index(anchor_img_ids[k])

    imshape = np.array([imgs.shape[2], imgs.shape[1]])
    featshape = np.array([feat_width, feat_height])
    anchor_uv_t = anchor_uv / imshape[None, None, None] * featshape[None, None, None]

    poses = camera_extrinsics[anchor_img_inds]

    query_xyz_cam = np.stack(
        [
            (np.linalg.inv(poses[i]) @ [*cur_query_coords[i], 1])[..., :3]
            for i in range(query_pts_per_batch)
        ],
        axis=0,
    )

    anchor_xyz_cam = np.zeros(
        (query_pts_per_batch, anchor_pts_per_query, imgs_per_anchor_pt, 3)
    )
    for i in range(query_pts_per_batch):
        for j in range(anchor_pts_per_query):
            anchor_xyz_cam[i, j] = (
                np.linalg.inv(poses[i, j]) @ [*anchor_xyz[i, j], 1]
            )[:, :3]

    rel_query_xyz = (query_xyz_cam - anchor_xyz_cam) / query_neighborhood_radius
    rel_query_xyz = positional_encoding(rel_query_xyz, L=10)
    rel_query_xyz = torch.Tensor(rel_query_xyz).cuda()

    sfm_xyz_cam = (np.linalg.inv(poses) @ np.c_[sfm_xyz, np.ones(len(sfm_xyz))].T)[
        ..., :3, :
    ].transpose(0, 1, 2, 4, 3)
    nn_sfm_xyz_cam = np.zeros(
        (
            query_pts_per_batch,
            anchor_pts_per_query,
            imgs_per_anchor_pt,
            n_pointnet_pts,
            3,
        )
    )
    for i in range(query_pts_per_batch):
        for j in range(anchor_pts_per_query):
            nn_sfm_xyz_cam[i, j] = sfm_xyz_cam[i, j][:, local_sfm_inds[i, j]]

    rel_nn_sfm_xyz = (
        nn_sfm_xyz_cam - anchor_xyz_cam[:, :, :, None, :]
    ) / query_neighborhood_radius

    rel_nn_sfm_xyz = positional_encoding(rel_nn_sfm_xyz, L=10)
    rel_nn_sfm_xyz = torch.Tensor(rel_nn_sfm_xyz).cuda()
    anchor_uv_t = torch.Tensor(anchor_uv_t).cuda()

    pointnet_feats, _ = torch.max(model["pointnet_mlp"](rel_nn_sfm_xyz), dim=3)

    shape = anchor_img_inds.shape
    _img_feats = img_feats[anchor_img_inds.flatten()].cuda()
    # cat_logits = model["img_classifier"](_img_feats[:, :32])

    pixel_feats = interp_maps(_img_feats, anchor_uv_t.reshape(-1, 2))
    pixel_feats = pixel_feats.reshape(*shape, -1)

    mlp_input = torch.cat((rel_query_xyz, pixel_feats, pointnet_feats), dim=3)
    query_pt_feats = model["mlp"](mlp_input)

    query_pt_feats, _ = torch.max(query_pt_feats, dim=1)
    query_pt_feats, _ = torch.max(query_pt_feats, dim=1)

    logits = model["occ_classifier"](query_pt_feats)[..., 0]

    """
    from mpl_toolkits.mplot3d import Axes3D
    f = figure()
    f.add_subplot(121, projection='3d')
    plot(sfm_xyz[:, 0], sfm_xyz[:, 1], sfm_xyz[:, 2], 'k.', markersize=0.1)
    plot(query_xyz[query_occ, 0], query_xyz[query_occ, 1], query_xyz[query_occ, 2], 'r.', markersize=1)
    plot(query_xyz[~query_occ, 0], query_xyz[~query_occ, 1], query_xyz[~query_occ, 2], 'b.', markersize=1)
    for i in range(len(anchor_xyz)):
        plot(anchor_xyz[i, :, 0], anchor_xyz[i, :, 1], anchor_xyz[i, :, 2], '.')

    n = int(floor(len(uv_t) ** .5))
    figure()
    for i in range(n ** 2):
        subplot(n, n, i + 1)
        imshow(imgs[visible_img_inds[i]])
        plot(uv[i, 0], uv[i, 1], 'r.')
        axis('off')
    tight_layout()

    figure()
    for i in range(n ** 2):
        subplot(n, n, i + 1)
        plot(uv_t.cpu()[i, 0], uv_t.cpu()[i, 1], 'r.')
        axis('off')
        xlim((0, img_feats.shape[3]))
        ylim((0, img_feats.shape[2]))
    tight_layout()

    from mpl_toolkits.mplot3d import Axes3D
    f = figure()
    f.add_subplot(121, projection='3d')
    plot(sfm_xyz[:, 0], sfm_xyz[:, 1], sfm_xyz[:, 2], '.', markersize=1)
    plot(poses[:, 0, 3], poses[:, 1, 3], poses[:, 2, 3], 'b.')
    plot([anchor_pt.xyz[0]], [anchor_pt.xyz[1]], [anchor_pt.xyz[2]], 'r.')

    sfm_xyz_cam = (
        np.linalg.inv(poses) @ np.c_[sfm_xyz, np.ones(len(sfm_xyz))].T
    ).transpose(2, 0, 1)

    f.add_subplot(122, projection='3d')
    plot(sfm_xyz_cam[:, 0, 0], sfm_xyz_cam[:, 0, 1], sfm_xyz_cam[:, 0, 2], '.', markersize=1)
    plot([anchor_xyz_cam[0, 0]], [anchor_xyz_cam[0, 1]], [anchor_xyz_cam[0, 2]], 'r.')
    plot([0], [0], [0], 'b.')
    """

    # query_embedding = model['query_embedder'](rel_query_xyz)
    # pointnet_input = torch.cat((pixel_feats, query_embedding), axis=2)
    # pointnet_input = torch.cat((pixel_feats, rel_query_xyz), axis=2)
    # logits = model['pointnet'](pointnet_input)[:, :, 0]
    preds = torch.sigmoid(logits)

    pred_volume[
        cur_query_inds[:, 0], cur_query_inds[:, 1], cur_query_inds[:, 2]
    ] = preds.cpu().numpy()

np.savez(
    "testmesh2.npz",
    verts=verts,
    faces=faces,
    pred_volume=pred_volume,
    pred_mask=pred_mask,
)


npz = load("testmesh2.npz")
verts = npz["verts"]
faces = npz["faces"]
pred_volume = npz["pred_volume"]
pred_mask = npz["pred_mask"]

import skimage.measure
import scipy.ndimage

verts, faces, _, _ = skimage.measure.marching_cubes(
    pred_volume, level=0.5, mask=scipy.ndimage.binary_erosion(pred_mask)
)


faces = np.concatenate((faces, faces[:, ::-1]), axis=0)
verts = verts[:, [1, 0, 2]]
verts = verts * res + minbounds

import open3d as o3d

mesh = o3d.geometry.TriangleMesh(
    o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces)
)
mesh.compute_triangle_normals()
mesh.compute_vertex_normals()

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
gt_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt_xyz[inds]))
gt_pcd.colors = o3d.utility.Vector3dVector(gt_rgb[inds])


o3d.visualization.draw_geometries([mesh, gt_pcd])
