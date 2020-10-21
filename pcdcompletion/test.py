import glob
import os

import numpy as np
import open3d as o3d
import skimage.measure
import torch

import config
import loader
import decoders
import pointnet


def predict_mesh(model, pts, rgb):
    minbounds = torch.min(pts, dim=0)[0].cpu().numpy()
    maxbounds = torch.max(pts, dim=0)[0].cpu().numpy()

    res = 0.16
    x = np.arange(minbounds[0], maxbounds[0], res)
    y = np.arange(minbounds[1], maxbounds[1], res)
    z = np.arange(minbounds[2], maxbounds[2], res)
    xx, yy, zz = np.meshgrid(x, y, z)
    query_coords = np.c_[xx.flatten(), yy.flatten(), zz.flatten()]
    query_vol = np.zeros(xx.shape)
    query_inds = np.argwhere(query_vol == 0)
    query_coords = torch.Tensor(query_coords[None]).cuda()
    grid_center = np.array([np.mean(y), np.mean(x), np.mean(z),])

    pointnet_inputs = torch.cat((pts, rgb), dim=-1)

    pointnet_feats = model["encoder"](pointnet_inputs[None])
    logits = model["decoder"](query_coords, None, pointnet_feats)
    preds = torch.sigmoid(logits).detach().cpu().numpy()

    query_vol[query_inds[:, 0], query_inds[:, 1], query_inds[:, 2]] = preds

    verts, faces, normals, values = skimage.measure.marching_cubes(query_vol, level=0.5)
    verts = (verts - np.mean(query_inds, axis=0)) * res + grid_center
    verts = verts[:, [1, 0, 2]]
    return verts, faces, preds, query_coords


scannet_dir = "/home/noah/data/scannet"
scan_dirs = sorted(glob.glob(os.path.join(scannet_dir, "*")))

# test_dset = loader.Dataset(scan_dirs[3:], 50, split="test")
test_dset = loader.Dataset(scan_dirs[3:], n_imgs=17, augment=True, load_gt_mesh=True)
test_loader = torch.utils.data.DataLoader(
    test_dset, batch_size=1, shuffle=False
)

encoder = pointnet.DumbPointnet(6, config.encoder_width)
decoder = decoders.Decoder(
    dim=3,
    z_dim=0,
    c_dim=config.encoder_width,
    hidden_size=config.decoder_width,
    leaky=False,
)
model = torch.nn.ModuleDict({"encoder": encoder, "decoder": decoder}).cuda()

checkpoint = torch.load("models/worthy-terrain-58")
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

pointnet_inputs = torch.cat((pts, rgb), dim=-1)
pointnet_feats = encoder(pointnet_inputs)
logits = decoder(query_coords, None, pointnet_feats)
loss = bce(logits[near_inds], query_occ[near_inds].float())

preds = torch.round(torch.sigmoid(logits))

n_examples = torch.sum(near_inds.float())
acc_num += torch.sum((preds.bool() == query_occ)[near_inds]).float().item()
loss_num += (loss * n_examples).item()
denom += n_examples.item()









j = 0
pts = pts[j].cuda()
rgb = rgb[j].cuda()









verts, faces, preds, query_coords = predict_mesh(model, pts, rgb)

pred_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(query_coords[0].cpu()))
pred_pcd.colors = o3d.utility.Vector3dVector(plt.cm.jet(preds[j])[:, :3])

pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.cpu()))
pcd.colors = o3d.utility.Vector3dVector(rgb.cpu()/255)

query_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(query_coords[j].cpu()))
query_pcd.colors = o3d.utility.Vector3dVector(plt.cm.jet(query_tsdf[j].cpu())[:, :3])

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

o3d.visualization.draw_geometries([gt_mesh], mesh_show_back_face=True)
o3d.visualization.draw_geometries([gt_mesh, mesh], mesh_show_back_face=True)
o3d.visualization.draw_geometries([pcd, mesh], mesh_show_back_face=True)
o3d.visualization.draw_geometries([pcd], mesh_show_back_face=True)
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
