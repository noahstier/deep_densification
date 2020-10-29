import glob
import os

import numpy as np
import open3d as o3d
import torch

import config
import common
import loader
import decoders
import pointnet

import train

scannet_dir = "/home/noah/data/scannet"
scan_dirs = sorted(glob.glob(os.path.join(scannet_dir, "*")))

test_dset = loader.Dataset(
    scan_dirs,
    load_gt_mesh=True,
    n_imgs=17,
    augment=False,
    # maxqueries=config.maxqueries,
    # maxpts=config.maxpts,
)
test_loader = torch.utils.data.DataLoader(test_dset, batch_size=1, shuffle=False)

model = train.Model.load_from_checkpoint(
    "models/pcd_completion/14cgh99h/checkpoints/epoch=210.ckpt"
)

model.eval()
model.requires_grad_(False)

batch = next(iter(test_loader))
(
    pts,
    rgb,
    query_coords,
    query_tsdf,
    gt_mesh_verts,
    gt_mesh_faces,
    gt_mesh_vertex_colors,
) = batch

query_occ = query_tsdf < 0

query_occ = query_occ
pts = pts
rgb = rgb
query_coords = query_coords
near_inds = query_tsdf < 1

logits = model(pts, rgb, query_coords)

loss = bce(logits[near_inds], query_occ[near_inds].float())

"""
j = 0
pt_inds = pts[j, :, 0] > -50
query_inds = query_tsdf[j] < 1
pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[j, pt_inds]))
pcd.colors = o3d.utility.Vector3dVector(rgb[j, pt_inds])
query_pcd = o3d.geometry.PointCloud(
    o3d.utility.Vector3dVector(query_coords[j, query_inds])
)
query_pcd.colors = o3d.utility.Vector3dVector(
    plt.cm.jet(query_tsdf[j, query_inds] * .5 + .5)[:, :3]
)
o3d.visualization.draw_geometries([pcd])
o3d.visualization.draw_geometries([pcd, query_pcd])

plot3()
plot(pts[j, pt_inds, 0], pts[j, pt_inds, 1], pts[j, pt_inds, 2], '.')
plot(query_coords[0, :, 0], query_coords[0, :, 1], query_coords[0, :, 2], '.')
"""

bce = torch.nn.BCEWithLogitsLoss()

loss = bce(logits[near_inds], query_occ[near_inds].float())

preds = torch.round(torch.sigmoid(logits))

n_examples = torch.sum(near_inds.float())
acc_num = torch.sum((preds.bool() == query_occ)[near_inds]).float().item()
loss_num = (loss * n_examples).item()
denom = n_examples.item()

test_loss = loss_num / denom
test_acc = acc_num / denom

j = 0
pts = pts[j]
rgb = rgb[j]

verts, faces, preds, query_coords = common.predict_mesh(model, pts, rgb)

pred_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(query_coords[0].cpu()))
pred_pcd.colors = o3d.utility.Vector3dVector(plt.cm.jet(preds[j])[:, :3])

pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.cpu()))
pcd.colors = o3d.utility.Vector3dVector(rgb.cpu())

query_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(query_coords[j].cpu()))
query_pcd.colors = o3d.utility.Vector3dVector(
    plt.cm.jet(0.5 + 0.5 * query_tsdf[j])[:, :3]
)

mesh = o3d.geometry.TriangleMesh(
    o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces)
)

gt_mesh = o3d.geometry.TriangleMesh(
    o3d.utility.Vector3dVector(gt_mesh_verts[j]),
    o3d.utility.Vector3iVector(gt_mesh_faces[j]),
)
gt_mesh.vertex_colors = o3d.utility.Vector3dVector(
    gt_mesh_vertex_colors[j].float() / 255
)

mesh.compute_vertex_normals()
gt_mesh.compute_vertex_normals()

o3d.visualization.draw_geometries([pred_pcd], mesh_show_back_face=True)
o3d.visualization.draw_geometries([gt_mesh, query_pcd], mesh_show_back_face=True)
o3d.visualization.draw_geometries([gt_mesh, mesh], mesh_show_back_face=True)
o3d.visualization.draw_geometries([pcd, mesh], mesh_show_back_face=True)
o3d.visualization.draw_geometries([pcd], mesh_show_back_face=True)
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
