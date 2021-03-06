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
import trimesh
import tqdm
import wandb

import fpn
import config
import unet

import depth_conditioned_coords

torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(1)


spec = importlib.util.spec_from_file_location(
    "colmap_reader", config.colmap_reader_script
)
colmap_reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(colmap_reader)


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

# included_classes = [5, 11, 20, 67, 72]
included_classes = [11]
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
img_ind = 79

rgb_imgdir = os.path.join(house_dir, "imgs/color")
depth_imgdir = os.path.join(house_dir, "imgs/depth")
cat_imgdir = os.path.join(house_dir, "imgs/category")
fusion_npz = np.load(os.path.join(house_dir, "fusion.npz"))
sfm_imfile = os.path.join(house_dir, "sfm/sparse/auto/images.bin")
sfm_ptfile = os.path.join(house_dir, "sfm/sparse/auto/points3D.bin")
delaunay_meshfile = os.path.join(house_dir, 'sfm/sparse/auto/meshed-delaunay.ply')

delaunay_mesh = o3d.io.read_triangle_mesh(delaunay_meshfile)
smoothed_mesh = delaunay_mesh.filter_smooth_laplacian()

ims = colmap_reader.read_images_binary(sfm_imfile)
pts = colmap_reader.read_points3d_binary(sfm_ptfile)

pts = {
    pt_id: pt for pt_id, pt in pts.items() if pt.error < 1 and len(pt.image_ids) >= 5
}

im_ids = sorted(ims.keys())
pt_ids = sorted(pts.keys())

sfm_xyz = np.stack([pts[i].xyz for i in pt_ids], axis=0)
sfm_rgb = np.stack([pts[i].rgb for i in pt_ids], axis=0) / 255

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
        "mlp": depth_conditioned_coords.MLP(),
        "cnn": fpn.FPN(input_height, input_width, 1),
    }
)
model.load_state_dict(torch.load("models/all-classes-1374134")["model"])
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

dense_pcd = o3d.io.read_point_cloud(os.path.join(house_dir, 'sfm2/dense/0/fused.ply'))
dense_xyz = np.asarray(dense_pcd.points)


cam_pos = camera_extrinsics[img_ind, :3, 3]
sfm_xyz_cam = (np.linalg.inv(camera_extrinsics[img_ind]) @ np.c_[sfm_xyz, np.ones(len(sfm_xyz))].T).T[:, :3]
sfm_uv = (camera_intrinsic @ sfm_xyz_cam.T).T
sfm_uv = sfm_uv[:, :2] / sfm_uv[:, 2:]
in_frustum_inds = (
    (sfm_xyz_cam[:, 2] > 0)
    & (sfm_uv[:, 0] >= 0)
    & (sfm_uv[:, 0] <= imwidth)
    & (sfm_uv[:, 1] >= 0)
    & (sfm_uv[:, 1] <= imheight)
)

verts = np.asarray(smoothed_mesh.vertices).astype(np.float64)
faces = np.asarray(smoothed_mesh.triangles).astype(np.int32)

dests = sfm_xyz
origins = np.tile(cam_pos[None], (len(dests), 1))
direcs = dests - origins
intersector = trimesh.ray.ray_triangle.RayMeshIntersector(trimesh.Trimesh(vertices=verts, faces=faces))
visible_inds = np.zeros(len(dests), dtype=bool)
for i in tqdm.trange(len(origins)):
    if in_frustum_inds[i]:
        locations, ray_idx, tri_idx = intersector.intersects_location(origins[i:i+1], direcs[i:i+1])
        intersect_dist = np.min(np.linalg.norm(locations - cam_pos, axis=1))
        pt_dist = np.linalg.norm(cam_pos - dests[i])
        if intersect_dist + .01 > pt_dist:
            visible_inds[i] = True


if False:
    # depth pixels, all classes
    anchor_inds = np.c_[uu.flatten(), vv.flatten()]
    anchor_uv = anchor_inds + 0.5
elif False:
    # depth pixels, included classes
    included_mask = np.sum([(cat_imgs[img_ind] == c) for c in included_classes], axis=0)
    anchor_inds = np.argwhere(included_mask > 0)[:, [1, 0]]
    anchor_inds = anchor_inds[
        np.random.choice(np.arange(len(anchor_inds)), size=500, replace=False)
    ]
    anchor_uv = anchor_inds + 0.5
elif False:
    # SFM points, all classes
    anchor_uv = visible_pt_uv
    anchor_inds = np.floor(anchor_uv).astype(int)
elif True:
    # estimated visible SFM points, all classes
    anchor_uv = sfm_uv[visible_inds]
    anchor_inds = np.floor(anchor_uv).astype(int)

anchor_xyz_cam = (
    np.linalg.inv(camera_intrinsic) @ np.c_[anchor_uv, np.ones(len(anchor_uv))].T
).T * depth_img[anchor_inds[:, 1], anchor_inds[:, 0], None]
anchor_xyz = (
    camera_extrinsics[img_ind] @ np.c_[anchor_xyz_cam, np.ones(len(anchor_xyz_cam))].T
).T[:, :3]


'''
anchor_xyz = sfm_xyz[visible_inds]
anchor_xyz_cam = (np.linalg.inv(camera_extrinsics[img_ind]) @ np.c_[anchor_xyz, np.ones(len(anchor_xyz))].T).T[:, :3]
'''

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

res = 0.02
maxbounds = np.max(anchor_xyz, axis=0) + 0.2
minbounds = np.min(anchor_xyz, axis=0) - 0.2
pred_vol_size = maxbounds - minbounds
n_bins = np.round(pred_vol_size / res).astype(int)

x = np.arange(-0.2, 0.2, res)
y = np.arange(-0.2, 0.2, res)
z = np.arange(-0.2, 0.2, res)
xx, yy, zz = np.meshgrid(x, y, z)
query_offsets = np.c_[xx.flatten(), yy.flatten(), zz.flatten()]
query_offsets = query_offsets[np.linalg.norm(query_offsets, axis=-1) < 0.2]

est_query_xyz = anchor_xyz[:, None] + query_offsets[None]
query_inds = np.round((est_query_xyz - minbounds) / res).astype(int)
query_xyz = query_inds * res + minbounds

img_feat = cnn(imgs_t[None, img_ind])

anchor_uv_t = (
    anchor_uv
    / [depth_img.shape[1], depth_img.shape[0]]
    * [img_feat.shape[3], img_feat.shape[2]]
)
pixel_feats = depth_conditioned_coords.interp_img(
    img_feat[0], torch.Tensor(anchor_uv_t).cuda()
).T

pred_vol = np.zeros(n_bins)
count_vol = np.zeros(n_bins, dtype=int)

for i in tqdm.trange(len(anchor_uv)):
    cur_query_xyz = query_xyz[i]
    cur_query_inds = query_inds[i]

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

    pred_vol[cur_query_inds[:, 0], cur_query_inds[:, 1], cur_query_inds[:, 2]] += preds
    count_vol[cur_query_inds[:, 0], cur_query_inds[:, 1], cur_query_inds[:, 2]] += 1

mean_pred_vol = np.zeros_like(pred_vol)
inds = count_vol > 0
mean_pred_vol[inds] = pred_vol[inds] / count_vol[inds]

verts, faces, _, _, = skimage.measure.marching_cubes(
    mean_pred_vol,
    level=0.5,
    mask=scipy.ndimage.morphology.binary_erosion(inds, iterations=2),
)
verts = verts - n_bins / 2
verts = verts * res + (maxbounds + minbounds) / 2
faces = np.concatenate((faces, faces[:, ::-1]), axis=0)

pred_mesh = o3d.geometry.TriangleMesh(
    o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces)
)
pred_mesh.compute_vertex_normals()
pred_mesh.compute_triangle_normals()

count_pts = argwhere(count_vol > 0)
count_pts = (count_pts - n_bins / 2) * res + (maxbounds + minbounds) / 2
count_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(count_pts))
counts = count_vol[count_vol > 0]
_counts = np.clip(counts, np.percentile(counts, 5), np.percentile(counts, 95))
_counts -= np.min(_counts)
_counts /= np.max(_counts)
count_pcd.colors = o3d.utility.Vector3dVector(plt.cm.jet(_counts)[:, :3])

pred_pts = argwhere(mean_pred_vol > 0)
pred_pts = (pred_pts - n_bins / 2) * res + (maxbounds + minbounds) / 2
pred_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pred_pts))
preds = mean_pred_vol[mean_pred_vol > 0]
pred_pcd.colors = o3d.utility.Vector3dVector(plt.cm.jet(preds)[:, :3])

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

# geoms = [pred_mesh, gt_pcd, anchor_spheres, cam_pcd]
geoms = [pred_mesh, gt_pcd]
# geoms = [pred_mesh]
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
