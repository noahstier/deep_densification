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
import unet
import wandb

import config
import depth_loader

torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(1)


def positional_encoding(xyz, L):
    encoding = []
    for l in range(L):
        encoding.append(np.sin(2 ** l ** np.pi * xyz))
        encoding.append(np.cos(2 ** l ** np.pi * xyz))
    encoding = np.concatenate(encoding, axis=-1)
    return encoding


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

        self.fc = torch.nn.Linear(k_in, k_out)
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

model = torch.nn.ModuleDict(
    {
        "query_encoder": torch.nn.Sequential(
            FCLayer(24, 32),
            FCLayer(32, 128),
            FCLayer(128, 256),
            FCLayer(256, 512),
            FCLayer(512, 1024),
        ),
        "mlp": torch.nn.Sequential(
            FCLayer(1280 + 1024, 1024),
            FCLayer(1024, 512),
            FCLayer(512, 256),
            FCLayer(256, 128),
            FCLayer(128, 64),
            FCLayer(64, 32),
            FCLayer(32, 16),
            torch.nn.Linear(16, 1),
        ),
        "cnn": torchvision.models.mobilenet_v2(pretrained=True).features,
    }
).cuda()

# model.load_state_dict(torch.load("models/depth-cond-global-87949"))
model.train()

if config.wandb:
    wandb.watch(model)

print("gathering house dirs")
house_dirs = sorted(
    [
        d
        for d in glob.glob(os.path.join(config.dset_dir, "*"))
        if os.path.exists(os.path.join(d, "fusion.npz"))
        and os.path.exists(os.path.join(d, "poses.npz"))
    ]
)
test_house_dirs = house_dirs[:10]
train_house_dirs = house_dirs[10:]
dset = depth_loader.Dataset(
    train_house_dirs,
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

optimizer = torch.optim.Adam(model.parameters())

occ_loss_fn = torch.nn.BCELoss().cuda()

step = -1
for epoch in range(20):
    print("epoch {}".format(epoch))

    for batch in tqdm.tqdm(loader):
        (
            query_coords,
            query_occ,
            query_uv,
            query_uv_t,
            query_xyz,
            query_xyz_cam,
            anchor_uv,
            anchor_uv_t,
            anchor_xyz_cam,
            rgb_img_t,
            rgb_img,
            depth_img,
            extrinsic,
            intrinsic,
        ) = batch

        if len(query_coords) < config.img_batch_size:
            raise Exception("gah")

        query_coords = torch.Tensor(positional_encoding(query_xyz.numpy(), L=4)).cuda()

        rgb_img_t = rgb_img_t.cuda()
        # query_coords = query_coords.cuda()
        query_occ = query_occ.float().cuda()
        query_uv_t = query_uv_t.cuda()

        optimizer.zero_grad()

        img_feats = torch.relu(model["cnn"](rgb_img_t)).mean([2, 3])

        query_coords = query_coords.reshape(
            config.img_batch_size, config.n_anchors * config.n_queries_per_anchor, 24
        )
        query_occ = query_occ.reshape(
            config.img_batch_size, config.n_anchors * config.n_queries_per_anchor
        )

        mlp_input = torch.cat(
            (
                model["query_encoder"](query_coords),
                img_feats[:, None].repeat(1, query_coords.shape[1], 1),
            ),
            dim=-1,
        )
        logits = model["mlp"](mlp_input)[..., 0]

        preds = torch.sigmoid(logits)

        pos_inds = query_occ.bool()
        neg_inds = ~pos_inds
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
                },
                step=step,
            )

    name = "depth-cond-global-{}".format(step)
    torch.save(model.state_dict(), os.path.join("models", name))


"""
j = 0
q = query_xyz_cam[j].cpu().numpy()
pred_inds = preds.detach().cpu()[j].round().bool()
gt_inds = query_occ.cpu()[j].round().bool()

gcf().add_subplot(121, projection='3d')
plot(q[pred_inds, 0], q[pred_inds, 1], q[pred_inds, 2], 'b.')
plot(q[~pred_inds, 0], q[~pred_inds, 1], q[~pred_inds, 2], 'r.')
plot([0], [0], [0], '.')

gcf().add_subplot(122, projection='3d')
plot(q[gt_inds, 0], q[gt_inds, 1], q[gt_inds, 2], 'b.')
plot(q[~gt_inds, 0], q[~gt_inds, 1], q[~gt_inds, 2], 'r.')
plot([0], [0], [0], '.')
"""
