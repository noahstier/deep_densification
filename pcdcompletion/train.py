import glob
import itertools
import os

import numpy as np
import open3d as o3d
import skimage.measure
import torch
import tqdm
import wandb

import loader
import decoders
import pointnet
import pointnet2

_wandb = True

scannet_dir = "/home/noah/data/scannet"
scan_dirs = sorted(glob.glob(os.path.join(scannet_dir, "*")))

train_dset = loader.Dataset(scan_dirs[3:], 10, split="train")
train_loader = torch.utils.data.DataLoader(
    train_dset, batch_size=3, shuffle=True, num_workers=8, drop_last=True
)

np.random.seed(0)
test_dset = loader.Dataset(scan_dirs[:3], 50, split="test")
test_batch = test_dset[0]

# encoder = pointnet.PointNetfeat(use_bn=False)
encoder = pointnet.DumbPointnet(6)
# encoder = pointnet2.models.PointNet2ClassificationMSG({"model.use_xyz": True})
decoder = decoders.Decoder(dim=3, z_dim=0, c_dim=1024, hidden_size=256, leaky=False)

model = torch.nn.ModuleDict({"encoder": encoder, "decoder": decoder}).cuda()

opt = torch.optim.Adam(model.parameters())
bce = torch.nn.BCEWithLogitsLoss()

if _wandb:
    wandb.init(project="pcd_completion")
    wandb.watch(model)

step = 0
for epoch in itertools.count():
    model.train()
    print("epoch {}".format(epoch))
    acc_num = 0
    loss_num = 0
    denom = 0
    for i, batch in enumerate(tqdm.tqdm(train_loader)):
        step += 1
        pts, rgb, query_coords, query_tsdf = batch

        """
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[0]))
        pcd.colors = o3d.utility.Vector3dVector(rgb[0])
        query_pcd = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(query_coords[0, query_tsdf[0] < 1])
        )
        query_pcd.colors = o3d.utility.Vector3dVector(
            plt.cm.jet(query_tsdf[0, query_tsdf[0] < 1])[:, :3]
        )
        o3d.visualization.draw_geometries([pcd, query_pcd])
        """

        query_occ = query_tsdf < 0

        query_occ = query_occ.cuda()
        pts = pts.cuda()
        rgb = rgb.cuda()
        query_coords = query_coords.cuda()

        pointnet_inputs = torch.cat((pts, rgb), dim=-1)

        opt.zero_grad()

        pointnet_feats = encoder(pointnet_inputs)

        near_inds = query_tsdf < 1

        logits = decoder(query_coords, None, pointnet_feats)

        loss = bce(logits[near_inds], query_occ[near_inds].float())

        loss.backward()
        opt.step()

        preds = torch.round(torch.sigmoid(logits))

        n_examples = torch.sum(near_inds.float())
        acc_num += torch.sum(preds.bool() == query_occ)[near_inds].float().item()
        loss_num += (loss * n_examples).item()
        denom += n_examples.item()

    if _wandb:
        wandb.log({"train loss": loss_num / denom, "train acc": acc_num / denom}, step=step)

    model.eval()
    (
        pts,
        rgb,
        query_coords,
        query_tsdf,
        gt_mesh_verts,
        gt_mesh_faces,
        gt_mesh_vertex_colors,
    ) = test_batch
    pts = torch.Tensor(pts).cuda()[None]
    rgb = torch.Tensor(rgb).cuda()[None]
    query_coords = torch.Tensor(query_coords).cuda()[None]
    query_tsdf = torch.Tensor(query_tsdf).cuda()[None]
    query_occ = query_tsdf < 0

    pointnet_inputs = torch.cat((pts, rgb), dim=-1)
    pointnet_feats = encoder(pointnet_inputs)
    logits = decoder(query_coords, None, pointnet_feats)
    loss = bce(logits, query_occ.float())
    preds = torch.round(torch.sigmoid(logits))
    acc = torch.mean((preds.bool() == query_occ).float())
    if _wandb:
        wandb.log({"test loss": loss.item(), "test acc": acc.item(),}, step=step)

    torch.save(
        {"model": model.state_dict(), "opt": opt.state_dict()},
        os.path.join("models", "test"),
    )
