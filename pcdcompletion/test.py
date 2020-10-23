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

scannet_dir = "/home/noah/data/scannet"
scan_dirs = sorted(glob.glob(os.path.join(scannet_dir, "*")))

test_dset = loader.Dataset(scan_dirs[:3], n_imgs=17, augment=True, load_gt_mesh=True)
test_loader = torch.utils.data.DataLoader(test_dset, batch_size=1, shuffle=False)

encoder = pointnet.DumbPointnet(6, config.encoder_width)
decoder = decoders.Decoder(
    dim=3,
    z_dim=0,
    c_dim=config.encoder_width,
    hidden_size=config.decoder_width,
    leaky=False,
)
model = torch.nn.ModuleDict({"encoder": encoder, "decoder": decoder}).cuda()

checkpoint = torch.load("models/dark-gorge-97")
model.load_state_dict(checkpoint["model"])

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

query_occ = query_occ.cuda()
pts = pts.cuda()
rgb = rgb.cuda()
query_coords = query_coords.cuda()
near_inds = query_tsdf < 1


pt_inds = pts[..., 0] > -50
pointnet_feats = []
for j in range(len(pt_inds)):
    pointnet_inputs = torch.cat((pts[j, pt_inds[j]], rgb[j, pt_inds[j]]), dim=-1)
    pointnet_feats.append(encoder(pointnet_inputs[None]))
pointnet_feats = torch.cat(pointnet_feats, dim=0)

logits = decoder(query_coords, None, pointnet_feats)
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

o3d.visualization.draw_geometries([gt_mesh, query_pcd], mesh_show_back_face=True)
o3d.visualization.draw_geometries([gt_mesh, mesh], mesh_show_back_face=True)
o3d.visualization.draw_geometries([pcd, mesh], mesh_show_back_face=True)
o3d.visualization.draw_geometries([pcd], mesh_show_back_face=True)
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
