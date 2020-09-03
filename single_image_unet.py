import functools
import glob
import importlib
import itertools
import os
import pickle

import imageio
import numpy as np
import open3d as o3d
import PIL.Image
import scipy.spatial
import torch
import torchvision
import tqdm
import unet
import wandb

import config
import single_image_loader

torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(1)


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
        "mlp": torch.nn.Sequential(
            FCLayer(1286, 256),
            FCLayer(256),
            FCLayer(256),
            FCLayer(256),
            FCLayer(256),
            FCLayer(256),
            FCLayer(256),
            FCLayer(256, 64),
            FCLayer(64, 16),
            torch.nn.Linear(16, 1),
        ),
        "cnn": torchvision.models.mobilenet_v2(pretrained=True).features,
        # "cnn": unet.UNet(3)
    }
).cuda()
# featheight, featwidth = model['cnn'](imgs_t[:1]).shape[2:]

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
dset = single_image_loader.Dataset(train_house_dirs, npts=config.pt_batch_size)
loader = torch.utils.data.DataLoader(
    dset, batch_size=config.img_batch_size, shuffle=True, num_workers=6, drop_last=True
)

optimizer = torch.optim.Adam(model.parameters())

occ_loss_fn = torch.nn.BCELoss().cuda()

step = -1
for epoch in range(20_000):
    print("epoch {}".format(epoch))

    # for batch in tqdm.tqdm(loader):
    for _ in itertools.count():
        batch = [dset[0] for _ in range(config.img_batch_size)]
        batch = [
            torch.Tensor(np.stack([a[i] for a in batch], axis=0)) for i in range(10)
        ]
        (
            query_coords,
            query_occ,
            query_uv,
            query_uv_t,
            query_xyz,
            query_xyz_cam,
            rgb_img_t,
            rgb_img,
            extrinsic,
            intrinsic,
        ) = batch

        rgb_img_t = rgb_img_t.cuda()
        query_coords = query_coords.cuda()
        query_occ = query_occ.float().cuda()
        query_uv_t = query_uv_t.cuda()

        optimizer.zero_grad()

        img_feats = torch.relu(model["cnn"](rgb_img_t))

        pixel_feats = img_feats.mean([2, 3])[:, None].repeat(
            1, query_coords.shape[1], 1
        )
        # pixel_feats = []
        # for i in range(len(img_feats)):
        #     pixel_feats.append(interp_img(img_feats[0], query_uv_t[0]).T)
        # pixel_feats = torch.stack(pixel_feats, dim=0).float()

        mlp_input = torch.cat((query_coords, pixel_feats), dim=2)
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

    name = "im-cond"
    torch.save(model.state_dict(), os.path.join("models", name))
