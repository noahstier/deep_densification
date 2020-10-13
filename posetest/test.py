import numpy as np
import open3d as o3d
import PIL.Image
import skimage.measure
import torch
import trimesh

import fpn


import pose_test_anchor
import loader


dset = loader.Dataset()

input_height, input_width = fpn.transform(PIL.Image.fromarray(dset[0][0])).shape[1:]

model = torch.nn.ModuleDict(
    {"cnn": fpn.FPN(input_height, input_width, 1), "mlp": pose_test_anchor.MLP(),}
).cuda()

checkpoint = torch.load("models/test-2500")
model.load_state_dict(checkpoint["model"])
model.eval()
model.requires_grad_(False)

res = 0.02
query_cube_size_len = 2 * dset.query_radius * np.sqrt(2) / 2
x = np.arange(-query_cube_size_len / 2, query_cube_size_len / 2, res)
y = np.arange(-query_cube_size_len / 2, query_cube_size_len / 2, res)
z = np.arange(-query_cube_size_len / 2, query_cube_size_len / 2, res)
query_offset_center = np.array([np.mean(x), np.mean(y), np.mean(z)])
xx, yy, zz = np.meshgrid(x, y, z)
query_offsets = np.c_[xx.flatten(), yy.flatten(), zz.flatten()]

(
    rgb_img,
    rgb_img_t,
    depth_img,
    anchor_uv,
    anchor_xyz_cam,
    anchor_xyz_cam_rotated,
    query_xyz_cam,
    query_xyz_cam_rotated,
    query_coords,
    query_occ,
    query_sd,
    pose,
    cam2anchor_rot,
    index,
) = dset[3]

j = 1
anchor_uv = anchor_uv[j]
anchor_xyz_cam = anchor_xyz_cam[j]
anchor_xyz_cam_rotated = anchor_xyz_cam_rotated[j]
query_xyz_cam = query_xyz_cam[j]
query_xyz_cam_rotated = query_xyz_cam_rotated[j]
query_coords = query_coords[j]
query_occ = query_occ[j]
query_sd = query_sd[j]
cam2anchor_rot = cam2anchor_rot[j]

fuze_trimesh = trimesh.load("fuze.obj")
v = np.asarray(fuze_trimesh.vertices)
vertmean = np.mean(v, axis=0)
diaglen = np.linalg.norm(np.max(v, axis=0) - np.min(v, axis=0))
v = (v - vertmean) / diaglen
v = (pose @ np.c_[v, np.ones(len(v))].T).T[:, :3]
fuze_trimesh.vertices = v

fuze_mesh = o3d.io.read_triangle_mesh("fuze.obj")
fuze_verts = np.asarray(fuze_mesh.vertices)
fuze_verts = (fuze_verts - vertmean) / diaglen
fuze_verts = (pose @ np.c_[fuze_verts, np.ones(len(fuze_verts))].T).T[:, :3]
fuze_mesh.vertices = o3d.utility.Vector3dVector(fuze_verts)


"""
subplot(221)
imshow(depth_img)
plot(anchor_uv[0], anchor_uv[1], '.')
gcf().add_subplot(222, projection='3d')
plot(query_xyz_cam[query_occ, 0], query_xyz_cam[query_occ, 1], query_xyz_cam[query_occ, 2], 'r.')
plot(query_xyz_cam[~query_occ, 0], query_xyz_cam[~query_occ, 1], query_xyz_cam[~query_occ, 2], 'b.')
plot([anchor_xyz_cam[0]], [anchor_xyz_cam[1]], [anchor_xyz_cam[2]], '.')
plot([0], [0], [0], '.')
xlabel('x')
ylabel('y')
gcf().add_subplot(223, projection='3d')
plot(query_xyz_cam_rotated[query_occ, 0], query_xyz_cam_rotated[query_occ, 1], query_xyz_cam_rotated[query_occ, 2], 'r.')
plot(query_xyz_cam_rotated[~query_occ, 0], query_xyz_cam_rotated[~query_occ, 1], query_xyz_cam[~query_occ, 2], 'b.')
plot([anchor_xyz_cam_rotated[0]], [anchor_xyz_cam_rotated[1]], [anchor_xyz_cam_rotated[2]], '.')
plot([0], [0], [0], '.')
xlabel('x')
ylabel('y')
gcf().add_subplot(224, projection='3d')
plot(query_coords[query_occ, 0], query_coords[query_occ, 1], query_coords[query_occ, 2], 'r.')
plot(query_coords[~query_occ, 0], query_coords[~query_occ, 1], query_coords[~query_occ, 2], 'b.')
plot([0], [0], [0], '.')
xlabel('x')
ylabel('y')
"""

rgb_img_t = torch.Tensor(rgb_img_t[None]).cuda()
query_coords = torch.Tensor(query_coords[None]).float().cuda()
query_occ = torch.Tensor(query_occ)
query_sd = torch.Tensor(query_sd).cuda()

img_feats = model["cnn"](rgb_img_t)
img_feats = torch.nn.functional.interpolate(
    img_feats, size=(rgb_img_t.shape[2:]), mode="bilinear", align_corners=False
)

anchor_uv_t = (
    anchor_uv
    / torch.Tensor([rgb_img.shape[1], rgb_img.shape[0]])
    * torch.Tensor([img_feats.shape[3], img_feats.shape[2]])
).cuda()

pixel_feats = pose_test_anchor.interp_img(img_feats[0], anchor_uv_t[None]).float().T
pixel_feats = pixel_feats[None].repeat((1, query_coords.shape[1], 1))

logits = model["mlp"](query_coords, pixel_feats)[0, ..., 0]
# preds = torch.sigmoid(logits)
preds = torch.tanh(logits)

"""
pred_pos = torch.round(preds).bool().cpu().numpy()
figure()
q = query_coords[0].cpu().numpy()
qo = query_occ.bool().numpy()
correct = qo == pred_pos
gcf().add_subplot(221, projection='3d')
plot(q[qo, 0], q[qo, 1], q[qo, 2], 'r.')
plot(q[~qo, 0], q[~qo, 1], q[~qo, 2], 'b.')
gcf().add_subplot(222, projection='3d')
plot(q[pred_pos, 0], q[pred_pos, 1], q[pred_pos, 2], 'r.')
plot(q[~pred_pos, 0], q[~pred_pos, 1], q[~pred_pos, 2], 'b.')
gcf().add_subplot(223, projection='3d')
plot(q[correct, 0], q[correct, 1], q[correct, 2], 'g.')
plot(q[~correct, 0], q[~correct, 1], q[~correct, 2], 'r.')

axes = o3d.geometry.TriangleMesh.create_coordinate_frame()

query_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(query_xyz_cam))
# query_pcd.colors = o3d.utility.Vector3dVector(
#     np.array([[0, 0, 1], [1, 0, 0]], dtype=float)[(preds.cpu().numpy() > 0).astype(int)]
# )
query_pcd.colors = o3d.utility.Vector3dVector(
    np.array([[0, 0, 1], [1, 0, 0]], dtype=float)[(query_sd.cpu().numpy() > 0).astype(int)]
)
# query_pcd.colors = o3d.utility.Vector3dVector(plt.cm.jet(query_sd.cpu().numpy() / dset.query_radius * .5 + .5)[:, :3])

cam_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
cam_mesh.paint_uniform_color(np.array([0, 0, 1], dtype=np.float64))
cam_mesh.compute_vertex_normals()
cam_mesh.compute_triangle_normals()

o3d.visualization.draw_geometries([query_pcd, axes, cam_mesh, fuze_mesh], mesh_show_back_face=True)
"""

pos_inds = query_sd > 0
true_pos = torch.sum((preds > 0) & pos_inds).float()
all_predicted_pos = torch.sum(preds > 0)
all_actual_pos = torch.sum(pos_inds)
precision = (true_pos / all_predicted_pos).item()
recall = (true_pos / all_actual_pos).item()


anchor_xyz_cam_rotated = (np.linalg.inv(cam2anchor_rot) @ anchor_xyz_cam.T).T
query_xyz_cam_rotated = anchor_xyz_cam_rotated + query_offsets

query_xyz_cam = (cam2anchor_rot @ query_xyz_cam_rotated.T).T

query_coords = (query_xyz_cam_rotated - anchor_xyz_cam_rotated) / dset.query_radius
# query_coords = pose_test_anchor.positional_encoding(query_coords, L=2)

query_coords = torch.Tensor(query_coords[None]).cuda()
pixel_feats = pose_test_anchor.interp_img(img_feats[0], anchor_uv_t[None]).float().T
pixel_feats = pixel_feats[None].repeat((1, query_coords.shape[1], 1))

logits = model["mlp"](query_coords, pixel_feats)[0, ..., 0]

preds = torch.tanh(logits).cpu().numpy()
pred_vol = np.reshape(preds, xx.shape)

verts, faces, _, _, = skimage.measure.marching_cubes(pred_vol, level=0)
faces = np.concatenate((faces, faces[:, ::-1]), axis=0)
verts = verts[:, [1, 0, 2]] - (np.array(pred_vol.shape) - 1) / 2
verts = verts * res + query_offset_center + anchor_xyz_cam_rotated
verts = (cam2anchor_rot @ verts.T).T

query_occ = fuze_trimesh.contains(query_xyz_cam)
query_sd = fuze_trimesh.nearest.signed_distance(query_xyz_cam)


"""
pred_pos = preds > .5
q = query_coords[0].cpu().numpy()
qo = query_occ
correct = qo == pred_pos
figure()
gcf().add_subplot(221, projection='3d')
plot(q[qo, 0], q[qo, 1], q[qo, 2], 'r.')
plot(q[~qo, 0], q[~qo, 1], q[~qo, 2], 'b.')
gcf().add_subplot(222, projection='3d')
plot(q[pred_pos, 0], q[pred_pos, 1], q[pred_pos, 2], 'r.')
plot(q[~pred_pos, 0], q[~pred_pos, 1], q[~pred_pos, 2], 'b.')
gcf().add_subplot(223, projection='3d')
plot(q[correct, 0], q[correct, 1], q[correct, 2], 'g.')
plot(q[~correct, 0], q[~correct, 1], q[~correct, 2], 'r.')
"""

true_pos = np.sum((preds > 0) & (query_sd > 0))
all_predicted_pos = np.sum(preds > 0)
all_actual_pos = np.sum(query_sd > 0)
precision = true_pos / all_predicted_pos
recall = true_pos / all_actual_pos

loss = np.abs(preds - query_sd / dset.query_radius)
inds = np.abs(query_sd) < 0.01
loss = np.sum(loss) / np.sum(inds)

"""
gcf().add_subplot(121, projection="3d")
plot(fuze_verts[:, 0], fuze_verts[:, 1], fuze_verts[:, 2], "k.")
gca().scatter(
    query_xyz_cam[:, 0],
    query_xyz_cam[:, 1],
    query_xyz_cam[:, 2],
    c=np.array([[0, 0, 1], [1, 0, 0]])[query_occ.astype(int)],
)
gcf().add_subplot(122, projection="3d")
plot(fuze_verts[:, 0], fuze_verts[:, 1], fuze_verts[:, 2], "k.")
gca().scatter(
    query_xyz_cam[:, 0],
    query_xyz_cam[:, 1],
    query_xyz_cam[:, 2],
    c=np.array([[0, 0, 1], [1, 0, 0]])[np.round(preds).astype(int)],
)


subplot(221)
imshow(depth_img)
plot(anchor_uv[0], anchor_uv[1], ".")
subplot(222)
imshow(rgb_img)
plot(anchor_uv[0], anchor_uv[1], ".")
gcf().add_subplot(223, projection="3d")
plot(fuze_verts[:, 0], fuze_verts[:, 1], fuze_verts[:, 2], ".")
plot(verts[:, 0], verts[:, 1], verts[:, 2], ".")
plot([0], [0], [0], ".")

"""

imshow(rgb_img)
plot(anchor_uv[0], anchor_uv[1], ".")

axes = o3d.geometry.TriangleMesh.create_coordinate_frame()

uv = np.argwhere(depth_img > 0)[:, [1, 0]]
ranges = depth_img[uv[:, 1], uv[:, 0]]
depth_reprojection = (dset.inv_intrinsic @ np.c_[uv, np.ones(len(uv))].T).T * -ranges[
    :, None
]
reproj_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(depth_reprojection))
reproj_pcd.paint_uniform_color(np.array([0, 1, 0], dtype=np.float64))

query_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(query_xyz_cam))
query_pcd.colors = o3d.utility.Vector3dVector(
    plt.cm.jet(query_sd / (2 * dset.query_radius) + 0.5)[:, :3]
)

inds = np.abs(query_sd) < 0.01
correct_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(query_xyz_cam[inds]))
correct_pcd.colors = o3d.utility.Vector3dVector(
    np.array([[1, 0, 0], [0, 1, 0]], dtype=float)[
        ((preds > 0) == (query_sd > 0)).astype(int)[inds]
    ]
)

cam_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
cam_mesh.paint_uniform_color(np.array([0, 0, 1], dtype=np.float64))
cam_mesh.compute_vertex_normals()
cam_mesh.compute_triangle_normals()

mesh = o3d.geometry.TriangleMesh(
    o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces)
)
mesh.compute_vertex_normals()
mesh.compute_triangle_normals()

# geoms = [fuze_mesh, query_pcd, axes, cam_mesh, mesh]
geoms = [fuze_mesh, query_pcd, mesh]
# geoms = [fuze_mesh, correct_pcd, mesh]
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
