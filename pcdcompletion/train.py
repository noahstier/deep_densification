import glob
import os

import numpy as np
import torch

import loader

def positional_encoding(xyz, L):
    encoding = []
    for l in range(L):
        encoding.append(torch.sin(2 ** l ** np.pi * xyz))
        encoding.append(torch.cos(2 ** l ** np.pi * xyz))
    encoding = torch.cat(encoding, axis=-1)
    return encoding

def mlp_block(n_in, n_out=None):
    if n_out is None:
        n_out = n_in
    return torch.nn.Sequential(
        torch.nn.Linear(n_in, n_out),
        torch.nn.ReLU(),
    )


class PointNet(torch.nn.Module):
    def __init__(self, in_feats):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            mlp_block(in_feats, 256),
            mlp_block(256),
            mlp_block(256),
            mlp_block(256),
            mlp_block(256),
            mlp_block(256, 1024),
        )

    def forward(self, x):
        x = self.mlp(x)
        x, _ = torch.max(x, dim=1)
        return x


scannet_dir = "/home/noah/data/scannet"
scan_dirs = sorted(glob.glob(os.path.join(scannet_dir, "*")))

dset = loader.Dataset(scan_dirs[:1], 10)
loader = torch.utils.data.DataLoader(dset, batch_size=1, shuffle=True, num_workers=8)

pointnet = PointNet(6).cuda()
classifier = torch.nn.Sequential(
    mlp_block(3 + pointnet.mlp[-1][0].out_features, 256),
    mlp_block(256),
    mlp_block(256),
    mlp_block(256),
    mlp_block(256),
    mlp_block(256),
    torch.nn.Linear(256, 1)
).cuda()

bce = torch.nn.BCEWithLogitsLoss()

opt = torch.optim.Adam([
    {'params': pointnet.parameters(), 'lr': .001},
    {'params': classifier.parameters(), 'lr': .001},
])


for batch in loader:
    break

pts, rgb, query_coords, query_tsdf = batch

query_occ = (query_tsdf < 0)

query_occ = query_occ.cuda()
pts = pts.cuda()
rgb = rgb.cuda()
query_coords = query_coords.cuda()
    
for i in range(1000):
    pointnet_inputs = torch.cat((pts, rgb), dim=-1)

    opt.zero_grad()

    pointnet_feats = pointnet(pointnet_inputs)
    
    pointnet_feats = pointnet_feats[:, None].repeat(1, query_coords.shape[1], 1)
    queries = torch.cat((pointnet_feats, query_coords), dim=-1)
    logits = classifier(queries)[..., 0]

    pos_loss = bce(logits[query_occ], query_occ[query_occ].float())
    neg_loss = bce(logits[~query_occ], query_occ[~query_occ].float())
    loss = pos_loss + neg_loss
    
    loss.backward()
    opt.step()

    preds = torch.round(torch.sigmoid(logits))
    pos_acc = torch.mean(preds[query_occ])
    neg_acc = torch.mean(1 - preds[~query_occ])

    print('{}, {:.2f}, {:.2f}, {:.2f}'.format(i, loss.item(), pos_acc.item(), neg_acc.item()))


pts, rgb, query_coords, query_tsdf = batch
pts = pts.cuda()
rgb = rgb.cuda()
query_coords = query_coords.cuda()

pointnet_inputs = torch.cat((pts, rgb), dim=-1)
pointnet_feats = pointnet(pointnet_inputs)
pointnet_feats = pointnet_feats[:, None].repeat(1, query_coords.shape[1], 1)
queries = torch.cat((pointnet_feats, query_coords), dim=-1)
logits = classifier(queries)[..., 0]
preds = torch.sigmoid(logits)

j = 0
xyz = query_coords[j].cpu().numpy()
pred_tsdf = preds[j].detach().cpu().numpy()
import open3d as o3d
import skimage.measure
query_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
query_pcd.colors = o3d.utility.Vector3dVector(plt.cm.jet(pred_tsdf)[:, :3])


geoms = [query_pcd]
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


vox = 

skimage.measure.marching_cubes(

