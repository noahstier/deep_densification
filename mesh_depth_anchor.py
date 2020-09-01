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

# house_dir = train_house_dirs[1]
# img_ind = 0
house_dir = test_house_dirs[-2]
img_ind = 39

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
        "mlp": MLP(),
        "cnn": fpn.FPN(input_height, input_width, 1),
    }
)
# model.load_state_dict(torch.load("models/sofa-only")['model'])
model.load_state_dict(torch.load("models/5-class-50hour")["model"])
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
x = np.arange(-0.2, 0.2, res)
y = np.arange(-0.2, 0.2, res)
z = np.arange(-0.2, 0.2, res)
xx, yy, zz = np.meshgrid(x, y, z)
query_offsets = np.c_[xx.flatten(), yy.flatten(), zz.flatten()]

x = np.arange(len(x), dtype=int)
y = np.arange(len(y), dtype=int)
z = np.arange(len(z), dtype=int)
xx, yy, zz = np.meshgrid(x, y, z)
query_inds = np.c_[xx.flatten(), yy.flatten(), zz.flatten()]

depth_img = depth_imgs[img_ind]
uu, vv = np.meshgrid(
    np.arange(50, depth_img.shape[1], 40), np.arange(50, depth_img.shape[0], 40)
)
# anchor_uv = np.c_[uu.flatten(), vv.flatten()]
included_mask = np.sum([(cat_imgs[img_ind] == c) for c in included_classes], axis=0)
anchor_uv = np.argwhere(included_mask > 0)[:, [1, 0]]
anchor_uv = anchor_uv[
    np.random.choice(np.arange(len(anchor_uv)), size=20, replace=False)
]
anchor_xyz_cam = (
    np.linalg.inv(camera_intrinsic) @ np.c_[anchor_uv, np.ones(len(anchor_uv))].T
).T * depth_img[anchor_uv[:, 1], anchor_uv[:, 0], None]
anchor_xyz = (
    camera_extrinsics[img_ind] @ np.c_[anchor_xyz_cam, np.ones(len(anchor_xyz_cam))].T
).T[:, :3]

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

query_xyz_cam_rotated = query_offsets[None] + anchor_xyz_cam_rotated[:, None]
query_xyz_cam = np.stack(
    [
        (cam2anchor_rot[i] @ query_xyz_cam_rotated[i].T).T
        for i in range(len(anchor_xyz_cam_rotated))
    ],
    axis=0,
)
query_uv = np.stack(
    [(camera_intrinsic @ query_xyz_cam[i].T).T for i in range(len(query_xyz_cam))]
)
query_uv = query_uv[..., :2] / query_uv[..., 2:]
query_xyz = np.stack(
    [
        (
            camera_extrinsics[img_ind]
            @ np.c_[query_xyz_cam[i], np.ones(len(query_xyz_cam[i]))].T
        ).T[:, :3]
        for i in range(len(query_xyz_cam))
    ],
    axis=0,
)
query_coords = torch.Tensor(
    (query_xyz_cam_rotated - anchor_xyz_cam_rotated[:, None]) / 0.2
).cuda()

img_feat, _ = cnn(imgs_t[None, img_ind])
anchor_uv_t = (
    anchor_uv
    / [depth_img.shape[1], depth_img.shape[0]]
    * [img_feat.shape[3], img_feat.shape[2]]
)
pixel_feats = interp_img(img_feat[0], torch.Tensor(anchor_uv_t).cuda()).T

anchors_per_batch = 24
for i in tqdm.trange(int(np.ceil(len(anchor_uv) / anchors_per_batch))):
    start = i * anchors_per_batch
    end = np.minimum(len(pixel_feats), (i + 1) * anchors_per_batch)

    logits = model["mlp"](
        query_coords[None, start:end],
        pixel_feats[None, start:end, None].repeat(1, 1, query_coords.shape[1], 1),
    )

    preds = torch.sigmoid(logits).cpu().numpy()[0, ..., 0]

    pred_vols[start:end, query_inds[:, 1], query_inds[:, 0], query_inds[:, 2]] = preds

"""
j = 4
imshow(imgs[img_ind])
plot(anchor_uv[j, 0], anchor_uv[j, 1], 'r.')
plot(query_uv[j, :, 0], query_uv[j, :, 1], 'b.', markersize=1)

imshow(imgs[img_ind])
plot(anchor_uv[:, 0], anchor_uv[:, 1], 'r.')

j = 4
verts, faces, _, _, = skimage.measure.marching_cubes(pred_vols[j], level=0.5)
verts = verts[:, [1, 0, 2]]
faces = np.concatenate((faces, faces[:, ::-1]), axis=0)
verts = (verts - np.array(xx.shape) / 2) * res + anchor_xyz_cam_rotated[j]
verts_cam = (cam2anchor_rot[j] @ verts.T).T
verts_world = (
    camera_extrinsics[img_ind] @ np.c_[verts_cam, np.ones(len(verts_cam))].T
).T[:, :3]

mesh = o3d.geometry.TriangleMesh(
    o3d.utility.Vector3dVector(verts_world), o3d.utility.Vector3iVector(faces)
)
mesh.compute_vertex_normals()
mesh.compute_triangle_normals()

pred_inds = np.round(pred_vols[j].flatten()).astype(np.bool)
pos_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(query_xyz[j, pred_inds]))
neg_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(query_xyz[j, ~pred_inds]))
pos_pcd.paint_uniform_color(np.array([0, 0, 1], dtype=np.float64))
neg_pcd.paint_uniform_color(np.array([1, 0, 0], dtype=np.float64))
o3d.visualization.draw_geometries([gt_pcd, mesh, pos_pcd, neg_pcd])
"""

meshes = []
for i in range(len(pred_vols)):
    verts, faces, _, _, = skimage.measure.marching_cubes(pred_vols[i], level=0.5)

    verts = verts[:, [1, 0, 2]]
    faces = np.concatenate((faces, faces[:, ::-1]), axis=0)

    verts = (verts - np.array(xx.shape) / 2) * res + anchor_xyz_cam_rotated[i]
    verts_cam = (cam2anchor_rot[i] @ verts.T).T
    verts_world = (
        camera_extrinsics[img_ind] @ np.c_[verts_cam, np.ones(len(verts_cam))].T
    ).T[:, :3]

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts_world), o3d.utility.Vector3iVector(faces)
    )
    # mesh.paint_uniform_color(np.array(plt.cm.jet(np.random.rand()))[:3])
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    meshes.append(mesh)


"""
pcd = o3d.geometry.PointCloud()
# for j in range(len(anchor_uv)):
for j in range(20):
    pred_inds = np.round(pred_vols[j].flatten()).astype(np.bool)
    pos_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(query_xyz[j, pred_inds]))
    neg_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(query_xyz[j, ~pred_inds]))
    pos_pcd.paint_uniform_color(np.array([0, 0, 1], dtype=np.float64))
    neg_pcd.paint_uniform_color(np.array([1, 0, 0], dtype=np.float64))
    pcd += pos_pcd
    pcd += neg_pcd
o3d.visualization.draw_geometries([pcd, *meshes[:20]])
o3d.visualization.draw_geometries([pcd])
"""


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

cam_pcd = o3d.geometry.PointCloud(
    o3d.utility.Vector3dVector(camera_extrinsics[img_ind : img_ind + 1, :3, 3])
)
cam_pcd.paint_uniform_color(np.array([1, 0, 0], dtype=np.float64))

# mesh = sum(meshes[::2])
anchor_inds = np.arange(len(meshes))
anchor_inds = np.random.choice(anchor_inds, size=10, replace=False)
# anchor_inds = np.array([ 4,  7,  3, 15,  2])

mesh = sum([meshes[i] for i in anchor_inds])

anchor_spheres = sum(
    [
        o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        .translate(anchor_xyz[i])
        .paint_uniform_color(np.array([0, 0, 1]))
        .compute_vertex_normals()
        for i in anchor_inds
    ]
)

geoms = [mesh, gt_pcd, anchor_spheres, cam_pcd]
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
