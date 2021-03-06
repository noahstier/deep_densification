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
import pyrender
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


def render_depth_img(mesh, extrinsic, intrinsic):
    fuze_trimesh = trimesh.Trimesh(
        np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    )
    mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
    scene = pyrender.Scene()
    scene.add(mesh)
    camera = pyrender.IntrinsicsCamera(
        intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
    )
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = extrinsic[:3, 3]
    camera_pose[:3, :3] = extrinsic[:3, :3]
    r = scipy.spatial.transform.Rotation.from_rotvec(
        np.array([1, 0, 0]) * np.pi
    ).as_matrix()
    camera_pose[:3, :3] = camera_pose[:3, :3] @ r
    scene.add(camera, pose=camera_pose)
    light = pyrender.SpotLight(
        color=np.ones(3),
        intensity=100,
        innerConeAngle=np.pi / 16.0,
        outerConeAngle=np.pi / 6.0,
    )
    scene.add(light, pose=camera_pose)
    r = pyrender.OffscreenRenderer(640, 480)
    _, depth = r.render(scene)
    return depth


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

rgb_imgdir = os.path.join(house_dir, "imgs/color")
depth_imgdir = os.path.join(house_dir, "imgs/depth")
cat_imgdir = os.path.join(house_dir, "imgs/category")
fusion_npz = np.load(os.path.join(house_dir, "fusion.npz"))
sfm_imfile = os.path.join(house_dir, "sfm/sparse/auto/images.bin")
sfm_ptfile = os.path.join(house_dir, "sfm/sparse/auto/points3D.bin")

delaunay_meshfile = os.path.join(house_dir, "sfm/sparse/auto/meshed-delaunay.ply")
delaunay_mesh = o3d.io.read_triangle_mesh(delaunay_meshfile)
smoothed_mesh = delaunay_mesh.filter_smooth_laplacian()

ims = colmap_reader.read_images_binary(sfm_imfile)
pts = colmap_reader.read_points3d_binary(sfm_ptfile)

pts = {
    pt_id: pt for pt_id, pt in pts.items() if pt.error < 1 and len(pt.image_ids) >= 5
}

im_ids = sorted(ims.keys())
pt_ids = np.array(sorted(pts.keys()))

sfm_xyz = np.stack([pts[i].xyz for i in pt_ids], axis=0).astype(np.float32)

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

print("loading images")
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

input_height = imgs_t.shape[2]
input_width = imgs_t.shape[3]

print("loading model")
model = torch.nn.ModuleDict(
    {
        "cnn": fpn.FPN(input_height, input_width, 1),
        "mlp": depth_conditioned_coords.MLP(),
    }
)
model.load_state_dict(torch.load("models/all-classes-392609")["model"])
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

res = 0.02
maxbounds = np.max(sfm_xyz, axis=0) + 0.2
minbounds = np.min(sfm_xyz, axis=0) - 0.2
pred_vol_size = maxbounds - minbounds
n_bins = np.round(pred_vol_size / res).astype(int)

x = np.arange(-0.2, 0.2, res)
y = np.arange(-0.2, 0.2, res)
z = np.arange(-0.2, 0.2, res)
xx, yy, zz = np.meshgrid(x, y, z)
query_offsets = np.c_[xx.flatten(), yy.flatten(), zz.flatten()]
query_offsets = query_offsets[np.linalg.norm(query_offsets, axis=-1) < 0.2]

est_query_xyz = sfm_xyz[:, None] + query_offsets[None]
query_inds = np.round((est_query_xyz - minbounds) / res).astype(int)
query_xyz = query_inds * res + minbounds

print("extracting image features")
img_feats = torch.cat([cnn(img_t[None]).cpu() for img_t in tqdm.tqdm(imgs_t)], dim=0)

pred_vol = np.zeros(n_bins)
count_vol = np.zeros(n_bins, dtype=int)

print("inference")
for im_id in tqdm.tqdm(im_ids[1::10]):
    im = ims[im_id]
    img_ind = im_id - 1

    sfm_xyz_cam = (
        np.linalg.inv(camera_extrinsics[img_ind])
        @ np.c_[sfm_xyz, np.ones(len(sfm_xyz))].T
    ).T[:, :3]

    sfm_uv = (camera_intrinsic @ sfm_xyz_cam.T).T
    sfm_uv = sfm_uv[:, :2] / sfm_uv[:, 2:]

    in_frustum_inds = (
        (sfm_xyz_cam[:, 2] > 0)
        & (sfm_uv[:, 0] >= 0)
        & (sfm_uv[:, 0] <= imwidth)
        & (sfm_uv[:, 1] >= 0)
        & (sfm_uv[:, 1] <= imheight)
    )

    est_depth_img = render_depth_img(
        smoothed_mesh, camera_extrinsics[img_ind], camera_intrinsic
    )

    sfm_depth = sfm_xyz_cam[:, 2]
    sfm_uv_inds = np.clip(
        np.round(sfm_uv).astype(int),
        [0, 0],
        [imwidth - 1, imheight - 1],
    )
    est_depth = est_depth_img[sfm_uv_inds[:, 1], sfm_uv_inds[:, 0]]
    est_visible_inds = (sfm_depth < est_depth + 0.01) & in_frustum_inds

    '''
    cam_pos = camera_extrinsics[img_ind, None, :3, 3]
    dests = sfm_xyz
    origins = np.tile(cam_pos, (len(dests), 1))
    direcs = dests - origins
    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(
        trimesh.Trimesh(vertices=verts, faces=faces)
    )
    visible_pt_inds = np.zeros(len(sfm_xyz), dtype=bool)
    for i in range(len(origins)):
        if in_frustum_inds[i]:
            locations, ray_idx, tri_idx = intersector.intersects_location(
                origins[i : i + 1], direcs[i : i + 1]
            )
            intersector.intersects_location(origins[i : i + 1], direcs[i : i + 1])
            intersect_dist = np.min(np.linalg.norm(locations - cam_pos, axis=1))
            pt_dist = np.linalg.norm(cam_pos - dests[i])
            if intersect_dist + 0.01 > pt_dist:
                visible_pt_inds[i] = True

    '''
    visible_pt_inds = np.argwhere(est_visible_inds).flatten()

    anchor_xyz = sfm_xyz[visible_pt_inds]
    anchor_xyz_cam = sfm_xyz_cam[visible_pt_inds]
    anchor_uv = sfm_uv[visible_pt_inds]

    """
    im_pt_ids = set(im.point3D_ids)
    visible_pt_inds = np.array(
        [i for i, pt_id in enumerate(pt_ids) if pt_id in im_pt_ids]
    )
    if len(visible_pt_inds) == 0:
        continue
    anchor_xyz = sfm_xyz[visible_pt_inds]
    """
    anchor_uv = np.clip(anchor_uv, [0, 0], [imgs.shape[2] - 1, imgs.shape[1] - 1])
    # anchor_uv = np.stack([xy for i, xy in enumerate(im.xys) if im.point3D_ids[i] in pts], axis=0)

    anchor_inds = np.floor(anchor_uv).astype(int)

    anchor_classes = cat_imgs[img_ind, anchor_inds[:, 1], anchor_inds[:, 0]]
    anchor_included = np.array([i in included_classes for i in anchor_classes])
    # anchor_included = np.ones(len(anchor_classes), dtype=bool)

    if np.sum(anchor_included) == 0:
        continue

    anchor_xyz = anchor_xyz[anchor_included]
    anchor_xyz_cam = anchor_xyz_cam[anchor_included]
    anchor_uv = anchor_uv[anchor_included]
    anchor_inds = anchor_inds[anchor_included]
    anchor_classes = anchor_classes[anchor_included]
    visible_pt_inds = visible_pt_inds[anchor_included]

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

    anchor_uv_t = (
        anchor_uv
        / [imgs.shape[2], imgs.shape[1]]
        * [img_feats.shape[3], img_feats.shape[2]]
    )

    pixel_feats = depth_conditioned_coords.interp_img(
        img_feats[img_ind], torch.Tensor(anchor_uv_t)
    ).T.cuda()
    pixel_feats = pixel_feats[:, None].repeat(1, query_xyz.shape[1], 1)

    for i in range(len(anchor_uv)):

        cur_query_xyz = query_xyz[visible_pt_inds[i]]
        cur_query_inds = query_inds[visible_pt_inds[i]]

        query_xyz_cam = (
            np.linalg.inv(camera_extrinsics[img_ind])
            @ np.c_[cur_query_xyz, np.ones(len(cur_query_xyz))].T
        ).T[:, :3]

        query_xyz_rotated = (np.linalg.inv(cam2anchor_rot[i]) @ query_xyz_cam.T).T
        anchor_xyz_rotated = np.linalg.inv(cam2anchor_rot[i]) @ anchor_xyz_cam[i]
        query_coords = (query_xyz_rotated - anchor_xyz_rotated) / 0.2
        query_coords = torch.Tensor(query_coords).cuda()

        logits = model["mlp"](query_coords[None], pixel_feats[i])

        preds = torch.sigmoid(logits)[0, ..., 0].cpu().numpy()

        pred_vol[
            cur_query_inds[:, 0], cur_query_inds[:, 1], cur_query_inds[:, 2]
        ] += preds
        count_vol[cur_query_inds[:, 0], cur_query_inds[:, 1], cur_query_inds[:, 2]] += 1

mean_pred_vol = np.zeros_like(pred_vol)
inds = count_vol > 2
mean_pred_vol[inds] = pred_vol[inds] / count_vol[inds]

verts, faces, _, _, = skimage.measure.marching_cubes(
    mean_pred_vol,
    level=0.5,
    mask=scipy.ndimage.morphology.binary_erosion(inds, iterations=2),
)
verts = (verts - n_bins / 2) * res + (maxbounds + minbounds) / 2
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
gt_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt_xyz))
gt_pcd.colors = o3d.utility.Vector3dVector(gt_rgb)

cam_pcd = o3d.geometry.PointCloud(
    o3d.utility.Vector3dVector(camera_extrinsics[img_ind : img_ind + 1, :3, 3])
)
cam_pcd.paint_uniform_color(np.array([1, 0, 0], dtype=np.float64))

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

# anchor_spheres = sum(
#     [
#         o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
#         .translate(sfm_xyz[i])
#         .paint_uniform_color(np.array([0, 0, 1]))
#         .compute_vertex_normals()
#         for i in range(len(sfm_xyz))
#     ]
# )
anchor_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sfm_xyz))
anchor_pcd.paint_uniform_color(np.array([0, 0, 1], dtype=np.float64))

# geoms = [pred_mesh, gt_pcd, anchor_pcd, cam_pcd, count_pcd]
# geoms = [pred_mesh, gt_pcd, anchor_pcd, cam_pcd, pred_pcd]
geoms = [pred_mesh, gt_pcd, anchor_pcd, cam_pcd]
# geoms = [pred_mesh]
# geoms = [count_pcd]
# geoms = [pred_mesh, pred_pcd]
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
