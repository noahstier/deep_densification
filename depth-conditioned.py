import functools
import glob
import importlib
import itertools
import os
import pickle

import open3d as o3d

import imageio
import numpy as np
import PIL.Image
import scipy.spatial
import torch
import torchvision
import tqdm
import wandb

import fpn
import unet
import config
import depth_loader

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)


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


def interp_img(img, xy):
    x = xy[:, 0]
    y = xy[:, 1]

    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()

    x0 = torch.clamp_min(x0, 0)
    y0 = torch.clamp_min(y0, 0)

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
        f_ll * ((x1 - x) * (y1 - y))
        + f_lr * ((x - x0) * (y1 - y))
        + f_ur * ((x - x0) * (y - y0))
        + f_ul * ((x1 - x) * (y - y0))
    )
    return interped


if config.wandb:
    wandb.init(project="deepmvs")

input_height, input_width = fpn.transform(
    PIL.Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
).shape[1:]

model = torch.nn.ModuleDict(
    {
        "query_encoder": torch.nn.Sequential(
            FCLayer(3, 64), FCLayer(64), FCLayer(64), FCLayer(64, 128),
        ),
        "mlp": torch.nn.Sequential(
            FCLayer(256),
            FCLayer(256, 512),
            FCLayer(512, 1024),
            FCLayer(1024),
            FCLayer(1024),
            FCLayer(1024, 512),
            FCLayer(512, 256),
            FCLayer(256, 64),
            FCLayer(64, 16),
            torch.nn.Linear(16, 1),
        ),
        # "cnn": torchvision.models.mobilenet_v2(pretrained=True).features[:7],
        # "cnn": unet.UNet(n_channels=3),
        "cnn": fpn.FPN(input_height, input_width, 1),
    }
).cuda()

optimizer = torch.optim.Adam(model.parameters())

if True:
    checkpoint = torch.load("models/5-class-128pt")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["opt"])


if config.wandb:
    wandb.watch(model)

house_dirs = sorted(
    [
        d
        for d in glob.glob(os.path.join(config.dset_dir, "*"))
        if os.path.exists(os.path.join(d, "fusion.npz"))
        and os.path.exists(os.path.join(d, "poses.npz"))
    ]
)

"""
"""
with open("per_img_classes.pkl", "rb") as f:
    per_img_classes = pickle.load(f)
"""
"""

included_classes = [5, 11, 20, 67, 72]
test_house_dirs = house_dirs[:10]
train_house_dirs = house_dirs[10:]
dset = depth_loader.Dataset(
    train_house_dirs,
    per_img_classes,
    included_classes,
    n_anchors=config.n_anchors,
    n_queries_per_anchor=config.n_queries_per_anchor,
)
loader = torch.utils.data.DataLoader(
    dset,
    batch_size=config.img_batch_size,
    shuffle=True,
    num_workers=8,
    drop_last=True,
    pin_memory=True,
)

occ_loss_fn = torch.nn.BCELoss(reduction="none").cuda()

step = -1
for epoch in range(10_000):
    print("epoch {}".format(epoch))

    model.train()
    for batch in tqdm.tqdm(loader):
        (
            query_coords,
            query_tsdf,
            query_uv,
            query_uv_t,
            query_xyz,
            query_xyz_cam,
            anchor_uv,
            anchor_uv_t,
            anchor_xyz_cam,
            anchor_classes,
            rgb_img_t,
            rgb_img,
            depth_img,
            extrinsic,
            intrinsic,
        ) = batch

        if len(query_coords) < config.img_batch_size:
            raise Exception("gah")

        rgb_img_t = rgb_img_t.cuda()
        query_coords = query_coords.cuda()
        query_tsdf = query_tsdf.float().cuda()
        query_uv_t = query_uv_t.cuda()
        query_occ = (query_tsdf <= 0).float()

        optimizer.zero_grad()

        img_feats, _ = model["cnn"](rgb_img_t)
        img_feats = torch.nn.functional.interpolate(
            img_feats, size=(rgb_img_t.shape[2:]), mode="bilinear", align_corners=False
        )

        anchor_uv_t_t = (
            anchor_uv_t
            / torch.Tensor([rgb_img_t.shape[3], rgb_img_t.shape[2]])
            * torch.Tensor([img_feats.shape[3], img_feats.shape[2]])
        ).cuda()

        pixel_feats = []
        for i in range(len(img_feats)):
            pixel_feats.append(interp_img(img_feats[i], anchor_uv_t_t[i]).T)
        pixel_feats = torch.stack(pixel_feats, dim=0).float()
        pixel_feats = pixel_feats[:, :, None].repeat((1, 1, query_coords.shape[2], 1))

        mlp_input = torch.cat(
            (model["query_encoder"](query_coords), pixel_feats), dim=-1
        )
        logits = model["mlp"](mlp_input)[..., 0]

        preds = torch.sigmoid(logits)

        pos_inds = query_occ.bool()
        neg_inds = ~pos_inds
        true_pos = (preds > 0.5) & pos_inds
        acc = torch.sum(true_pos) / torch.sum(pos_inds).float()
        class_acc = {}
        for c in included_classes:
            inds = anchor_classes == c
            denom = torch.sum(pos_inds[inds]).float()
            if denom > 0:
                class_acc[c] = torch.sum(true_pos[inds]) / denom

        pos_acc = torch.sum((preds > 0.5) & pos_inds) / pos_inds.sum().float()
        neg_acc = torch.sum((preds < 0.5) & neg_inds) / neg_inds.sum().float()

        true_pos = torch.sum((preds > 0.5) & pos_inds).float()
        all_predicted_pos = torch.sum(preds > 0.5)
        all_actual_pos = torch.sum(pos_inds)
        precision = true_pos / all_predicted_pos
        recall = true_pos / all_actual_pos

        # occ_loss = occ_loss_fn(preds, query_occ)
        # cat_loss = cat_loss_fn(cat_logits, cat_gt[visible_img_inds])

        pos_loss = occ_loss_fn(preds[pos_inds], query_occ[pos_inds])
        neg_loss = occ_loss_fn(preds[neg_inds], query_occ[neg_inds])

        # loss = pos_loss + neg_loss
        loss = occ_loss_fn(preds, query_occ)
        loss = torch.mean(loss)

        """
        i = 1
        j = 3

        x = y = z = np.arange(-1, 1, .2)
        xx, yy, zz = np.meshgrid(x, y, z)
        xyz = np.c_[xx.flatten(), yy.flatten(), zz.flatten()]
        fake_query_coords = torch.Tensor(xyz).cuda()[None, None]


        pixel_feat = pixel_feats[i, j, 0][None, None, None, :].repeat((1, 1, fake_query_coords.shape[2], 1))

        mlp_input = torch.cat((model["query_encoder"](fake_query_coords), pixel_feat), dim=-1)
        fake_logits = model["mlp"](mlp_input)[..., 0]
        fake_preds = torch.sigmoid(fake_logits)

        subplot(221)
        imshow(rgb_img[i].numpy())
        plot(anchor_uv[i, j, 0], anchor_uv[i, j, 1], '.')
        axis('off')

        gcf().add_subplot(222, projection='3d')
        q = query_coords.cpu().numpy()
        t = query_tsdf.cpu().numpy()
        gca().scatter(q[i, j, :, 0], q[i, j, :, 1], q[i, j, :, 2], alpha=1, c=plt.cm.jet(1 - (t[i, j] * .5 + .5))[:, :3])
        plot([0], [0], [0], 'k.', markersize=10)
        xlabel('x')
        ylabel('y')

        gcf().add_subplot(223, projection='3d')
        q = query_coords.cpu().numpy()
        pr = preds.detach().cpu().numpy()
        gca().scatter(q[i, j, :, 0], q[i, j, :, 1], q[i, j, :, 2], alpha=1, c=plt.cm.jet(pr[i, j])[:, :3])
        plot([0], [0], [0], 'k.', markersize=10)
        xlabel('x')
        ylabel('y')

        gcf().add_subplot(224, projection='3d')
        q = fake_query_coords.cpu().numpy()
        fpr = fake_preds.detach().cpu().numpy()
        gca().scatter(q[0, 0, :, 0], q[0, 0, :, 1], q[0, 0, :, 2], alpha=1, c=plt.cm.jet(fpr[0, 0])[:, :3])
        plot([0], [0], [0], 'k.', markersize=10)
        xlabel('x')
        ylabel('y')

        verts, faces, _, _, = skimage.measure.marching_cubes(fpr[0, 0].reshape(xx.shape), level=0.5)
        verts = verts[:, [1, 0, 2]]
        verts = (verts - np.array(xx.shape) / 2) * .2
        faces = np.concatenate((faces, faces[:, ::-1]), axis=0)
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces))
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(q[0, 0]))
        pcd.colors = o3d.utility.Vector3dVector(plt.cm.jet(fpr[0, 0])[:, :3])
        o3d.visualization.draw_geometries([mesh, pcd])
        """

        loss.backward()
        optimizer.step()

        step += 1
        if config.wandb:
            wandb.log(
                {
                    "pos loss": pos_loss.item(),
                    "neg loss": neg_loss.item(),
                    # "loss": loss.item(),
                    # "cat_loss": cat_loss.item(),
                    "logits": wandb.Histogram(logits.detach().cpu().numpy()),
                    "preds": wandb.Histogram(preds.detach().cpu().numpy()),
                    "occ": wandb.Histogram(query_occ.cpu().numpy()),
                    "pos_acc": pos_acc.item(),
                    "neg_acc": neg_acc.item(),
                    "precision": precision.item(),
                    "recall": recall.item(),
                    **{
                        "class acc {}".format(str(k)): v.item()
                        for k, v in class_acc.items()
                    },
                },
                step=step,
            )

    name = "depth-cond-anchor-{}".format(step)
    torch.save(
        {"model": model.state_dict(), "opt": optimizer.state_dict()},
        os.path.join("models", name),
    )

"""
j = 0
q = query_xyz_cam[j].cpu().numpy().reshape(-1, 3)
pred_inds = preds.detach().cpu()[j].round().bool().flatten()
gt_inds = query_occ.cpu()[j].round().bool().flatten()

gcf().add_subplot(131, projection='3d')
plot(q[pred_inds, 0], q[pred_inds, 1], q[pred_inds, 2], 'b.')
plot(q[~pred_inds, 0], q[~pred_inds, 1], q[~pred_inds, 2], 'r.')
plot([0], [0], [0], '.')
title('pred occ')

gcf().add_subplot(132, projection='3d')
plot(q[gt_inds, 0], q[gt_inds, 1], q[gt_inds, 2], 'b.')
plot(q[~gt_inds, 0], q[~gt_inds, 1], q[~gt_inds, 2], 'r.')
plot([0], [0], [0], '.')
title('gt occ')

gcf().add_subplot(133, projection='3d')
plot(q[gt_inds == pred_inds, 0], q[gt_inds == pred_inds, 1], q[gt_inds == pred_inds, 2], 'b.')
plot(q[gt_inds != pred_inds, 0], q[gt_inds != pred_inds, 1], q[gt_inds != pred_inds, 2], 'r.')
plot([0], [0], [0], '.')
title('pred == gt')
"""
