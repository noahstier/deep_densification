import datetime
import glob
import itertools
import os

import numpy as np
import open3d as o3d
import skimage.measure
import torch
import tqdm
import wandb

import common
import config
import loader
import decoders
import pointnet
import pointnet2

"""
todo:

    - learn with translation
    - learn with augmentation
    - expand dataset

to improve:
    more scans
    cbatchnorm?
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

train_dirs = [
    scan_dirs[0],
    scan_dirs[0],
    scan_dirs[0],
    scan_dirs[0],
    scan_dirs[3],
    scan_dirs[3],
    scan_dirs[3],
    scan_dirs[3],
]
test_dirs = [
    scan_dirs[0],
    scan_dirs[3],
]

train_dset = loader.Dataset(train_dirs, n_imgs=17, augment=True)
train_loader = torch.utils.data.DataLoader(
    train_dset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=config.num_workers,
    drop_last=True,
    worker_init_fn=lambda worker_id: np.random.seed(),
)

test_dset = loader.Dataset(test_dirs, n_imgs=17, augment=False)
test_batch = test_dset[0]
test_loader = torch.utils.data.DataLoader(
    test_dset,
    batch_size=1,
    shuffle=False,
    num_workers=2,
    drop_last=False,
    worker_init_fn=lambda worker_id: np.random.seed(),
)

# encoder = pointnet.DumbPointnet(6, config.encoder_width)
encoder = pointnet.DumbPointnet(6, config.encoder_width)
decoder = decoders.DecoderBatchNorm(
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
    checkpoint = torch.load("models/summer-sea-92")
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
        o3d.visualization.draw_geometries([pcd])
        o3d.visualization.draw_geometries([pcd, query_pcd])
        """

        query_occ = query_tsdf < 0

        query_occ = query_occ.cuda()
        pts = pts.cuda()
        rgb = rgb.cuda()
        query_coords = query_coords.cuda()
        near_inds = query_tsdf < 1

        opt.zero_grad()

        pt_inds = pts[..., 0] > -50
        pointnet_feats = []
        for j in range(len(pt_inds)):
            pointnet_inputs = torch.cat((pts[j, pt_inds[j]], rgb[j, pt_inds[j]]), dim=-1)
            pointnet_feats.append(encoder(pointnet_inputs[None]))
        pointnet_feats = torch.cat(pointnet_feats, dim=0)

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
        o3d.visualization.draw_geometries([pcd])
        """

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
        loss = bce(logits[near_inds], query_occ[near_inds].float())

        preds = torch.round(torch.sigmoid(logits))

        n_examples = torch.sum(near_inds.float())
        acc_num += torch.sum((preds.bool() == query_occ)[near_inds]).float().item()
        loss_num += (loss * n_examples).item()
        denom += n_examples.item()

    test_loss = loss_num / denom
    test_acc = acc_num / denom

    wandb_logs = {}

    pts, rgb, _, _ = test_batch
    pts = torch.Tensor(pts).cuda()
    rgb = torch.Tensor(rgb).cuda()
    try:
        verts, faces, preds, query_coords = common.predict_mesh(model, pts, rgb)
        test_mesh_img = common.render_mesh(verts, faces)
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces)
        )
        mesh.compute_vertex_normals()
        mesh_dir = os.path.join('meshes', model_name)
        os.makedirs(mesh_dir, exist_ok=True)
        o3d.io.write_triangle_mesh(os.path.join(mesh_dir, str(step).zfill(6) + '.ply'), mesh)
        wandb_logs['test/mesh'] = wandb.Image(test_mesh_img)
    except Exception as e:
        print('meshing failed', e)

    if config.wandb:
        wandb.log(
            {
                **wandb_logs,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "test/loss": test_loss,
                "test/acc": test_acc,
            },
            step=step,
        )

    torch.save(
        {"model": model.state_dict(), "opt": opt.state_dict()},
        os.path.join("models", model_name)
    )
