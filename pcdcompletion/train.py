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
import pointnetpp

"""
to improve:
    more scans
    cbatchnorm?
    image feats
    pointnet++

unstable test performance?
    - tune batch norm momentum. try .1 or .5
    - normalization: divide by std, divide by 5, normalize to unit sphere, no norm
"""


def train_loop(model, loader, opt):
    model.train()
    acc_num = 0
    loss_num = 0
    denom = 0
    n_steps = 0
    for i, batch in enumerate(tqdm.tqdm(loader)):
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

        logits = model(pts, rgb, query_coords)

        loss = bce(logits[near_inds], query_occ[near_inds].float())

        loss.backward()
        opt.step()
        n_steps += 1

        preds = torch.round(torch.sigmoid(logits))

        n_examples = torch.sum(near_inds.float())
        acc_num += torch.sum((preds.bool() == query_occ)[near_inds]).float().item()
        loss_num += (loss * n_examples).item()
        denom += n_examples.item()

    avg_loss = loss_num / denom
    avg_acc = acc_num / denom

    return n_steps, avg_loss, avg_acc


def test_loop(model, loader):
    model.eval()

    acc_num = 0
    loss_num = 0
    denom = 0
    for i, batch in enumerate(tqdm.tqdm(loader)):
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

        logits = model(pts, rgb, query_coords)
        loss = bce(logits[near_inds], query_occ[near_inds].float())

        preds = torch.round(torch.sigmoid(logits))

        n_examples = torch.sum(near_inds.float())
        acc_num += torch.sum((preds.bool() == query_occ)[near_inds]).float().item()
        loss_num += (loss * n_examples).item()
        denom += n_examples.item()

    avg_loss = loss_num / denom
    avg_acc = acc_num / denom
    return avg_loss, avg_acc


scan_dirs = sorted(glob.glob(os.path.join(config.scannet_dir, "*")))

train_dset = loader.Dataset(
    scan_dirs[:8],
    n_imgs=17,
    augment=True,
    maxqueries=config.maxqueries,
    maxpts=config.maxpts,
)
train_loader = torch.utils.data.DataLoader(
    train_dset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=config.num_workers,
    drop_last=True,
    worker_init_fn=lambda worker_id: np.random.seed(),
)

test_dset = loader.Dataset(
    scan_dirs[:8],
    n_imgs=17,
    augment=False,
    maxqueries=config.maxqueries,
    maxpts=config.maxpts,
)
test_batch = test_dset[0]
test_loader = torch.utils.data.DataLoader(
    test_dset,
    batch_size=1,
    shuffle=False,
    num_workers=2,
    drop_last=False,
    worker_init_fn=lambda worker_id: np.random.seed(),
)


def percentile(t, q):
    k = 1 + round(0.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k).values.item()
    return result


class Model(torch.nn.Module):
    def __init__(self, pt_dim=3, encoder_width=1024, decoder_width=256, bnm=0.1):
        super().__init__()
        self.encoder = pointnet.DumbPointnet(6, encoder_width)  # , bnm=config.bnm)
        self.decoder = decoders.DecoderBatchNorm(
            dim=3,
            z_dim=0,
            c_dim=encoder_width,
            hidden_size=decoder_width,
            leaky=False,
            bnm=bnm,
        )

    def forward(self, pts, rgb, query_coords):
        pt_inds = pts[..., 0] > -50
        pointnet_feats = []
        for j in range(len(pt_inds)):
            norm_99_pct = percentile(torch.norm(pts[j, pt_inds[j]], dim=-1), 99)
            pts[j] /= norm_99_pct
            query_coords[j] /= norm_99_pct

            pointnet_inputs = torch.cat(
                (pts[j, pt_inds[j]], rgb[j, pt_inds[j]]), dim=-1
            )
            pointnet_feats.append(self.encoder(pointnet_inputs[None]))
        pointnet_feats = torch.cat(pointnet_feats, dim=0)
        logits = self.decoder(query_coords, None, pointnet_feats)
        return logits


model = Model(
    pt_dim=6,
    encoder_width=config.encoder_width,
    decoder_width=config.decoder_width,
    bnm=config.bnm,
).cuda()


opt = torch.optim.Adam(model.parameters())
bce = torch.nn.BCEWithLogitsLoss()

if False:
    checkpoint = torch.load("models/summer-sea-92")
    model.load_state_dict(checkpoint["model"])

if config.wandb:
    wandb.init(project="pcd_completion")
    wandb.watch(model)
    model_name = wandb.run.name
else:
    model_name = str(datetime.datetime.now()).replace(" ", "_")

step = 0
for epoch in itertools.count():
    print("epoch {}".format(epoch))

    n_steps, train_loss, train_acc = train_loop(model, train_loader, opt)
    step += n_steps

    test_loss, test_acc = test_loop(model, test_loader)

    wandb_logs = {
        "train/loss": np.round(train_loss, 3),
        "train/acc": np.round(train_acc, 3),
        "test/loss": np.round(test_loss, 3),
        "test/acc": np.round(test_acc, 3),
    }

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
        mesh_dir = os.path.join("meshes", model_name)
        os.makedirs(mesh_dir, exist_ok=True)
        o3d.io.write_triangle_mesh(
            os.path.join(mesh_dir, str(step).zfill(6) + ".ply"), mesh
        )
        wandb_logs["test/mesh"] = wandb.Image(test_mesh_img)
    except Exception as e:
        print("meshing failed", e)

    if config.wandb:
        wandb.log(wandb_logs, step=step)
    else:
        print(wandb_logs)

    torch.save(
        {"model": model.state_dict(), "opt": opt.state_dict()},
        os.path.join("models", model_name),
    )
