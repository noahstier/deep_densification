import glob
import itertools
import os

import numpy as np
import open3d as o3d
import skimage.measure
import torch
import tqdm
import wandb

import config
import loader
import decoders
import pointnet
import pointnet2

"""

todo:
    - divide by variance inside model: array([1.37907848, 2.33269393, 0.40424237])
    - reduce initialization sensitivity: batchnorm?
    - get test acc to match train acc during overfitting


to improve:
    more scans
    cbatchnorm
    image feats
    pointnet++
"""

scan_dirs = sorted(
    [
        d
        for d in glob.glob(os.path.join(config.scannet_dir, "*"))
        if os.path.exists(os.path.join(d, "tsdf_0.16.npz"))
    ]
)

train_dset = loader.Dataset(scan_dirs[3:], n_imgs=17, augment=True)
train_loader = torch.utils.data.DataLoader(
    train_dset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=config.num_workers,
    drop_last=True,
)

test_dset = loader.Dataset(scan_dirs[3:6], n_imgs=17, augment=False)
test_loader = torch.utils.data.DataLoader(
    test_dset,
    batch_size=1,
    shuffle=False,
    num_workers=2,
    drop_last=False,
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

opt = torch.optim.Adam(model.parameters())
bce = torch.nn.BCEWithLogitsLoss()

if config.wandb:
    wandb.init(project="pcd_completion")
    wandb.watch(model)
    model_name = wandb.run.name
else:
    model_name = str(datetime.datetime.now()).replace(' ', '_')


if False:
    checkpoint = torch.load("models/test")
    model.load_state_dict(checkpoint["model"])

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
        o3d.visualization.draw_geometries([pcd, query_pcd])
        """

        query_occ = query_tsdf < 0

        query_occ = query_occ.cuda()
        pts = pts.cuda()
        rgb = rgb.cuda()
        query_coords = query_coords.cuda()
        near_inds = query_tsdf < 1

        pointnet_inputs = torch.cat((pts, rgb), dim=-1)

        opt.zero_grad()

        pointnet_feats = encoder(pointnet_inputs)

        logits = decoder(query_coords, None, pointnet_feats)

        loss = bce(logits[near_inds], query_occ[near_inds].float())

        loss.backward()
        opt.step()

        preds = torch.round(torch.sigmoid(logits))

        n_examples = torch.sum(near_inds.float())
        acc_num += torch.sum((preds.bool() == query_occ)[near_inds]).float().item()
        loss_num += (loss * n_examples).item()
        denom += n_examples.item()

    train_loss = loss_num / denom
    train_acc = acc_num / denom

    model.eval()

    acc_num = 0
    loss_num = 0
    denom = 0
    for i, batch in enumerate(tqdm.tqdm(test_loader)):
        pts, rgb, query_coords, query_tsdf = batch

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
        o3d.visualization.draw_geometries([pcd, query_pcd])
        """

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

    test_loss = loss_num / denom
    test_acc = acc_num / denom

    if config.wandb:
        wandb.log(
            {
                "train loss": train_loss,
                "train acc": train_acc,
                "test loss": test_loss,
                "test acc": test_acc,
            },
            step=step,
        )

    torch.save(
        {"model": model.state_dict(), "opt": opt.state_dict()},
        os.path.join("models", model_name)
    )
