import glob
import logging
import os

import cv2
import h5py
import matplotlib.pyplot as plt
import PIL.Image
import numpy as np
import open3d as o3d
import scipy.spatial
import torch
import torchvision
import trimesh
import tqdm
import wandb

import config

trimesh.constants.log.setLevel(logging.ERROR)

import fpn
import loader


torch.manual_seed(1)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
np.random.seed(1)


def log_transform(x, shift=1):
    return x.sign() * (1 + x.abs() / shift).log()


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


def positional_encoding(xyz, L):
    encoding = []
    for l in range(L):
        encoding.append(np.sin(2 ** l ** np.pi * xyz))
        encoding.append(np.cos(2 ** l ** np.pi * xyz))
    encoding = np.concatenate(encoding, axis=-1)
    return encoding


def bn_relu_fc(in_c, out_c):
    return torch.nn.Sequential(
        torch.nn.BatchNorm1d(in_c),
        torch.nn.ReLU(),
        torch.nn.Linear(in_c, out_c, bias=False),
    )


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.coord_encoder = torch.nn.Sequential(
            torch.nn.Linear(3, 32, bias=False),
            bn_relu_fc(32, 32),
            bn_relu_fc(32, 64),
            bn_relu_fc(64, 128),
            bn_relu_fc(128, 256),
        )

        self.classifier = torch.nn.Sequential(
            bn_relu_fc(512, 512),
            bn_relu_fc(512, 256),
            bn_relu_fc(256, 256),
            bn_relu_fc(256, 256),
            bn_relu_fc(256, 256),
            bn_relu_fc(256, 256),
            bn_relu_fc(256, 256),
            bn_relu_fc(256, 64),
            bn_relu_fc(64, 16),
            bn_relu_fc(16, 1),
        )

    def forward(self, coords, feats):
        shape = coords.shape[:-1]
        coords = coords.reshape(np.prod(list(shape)), coords.shape[-1])
        feats = feats.reshape(np.prod(list(shape)), feats.shape[-1])

        encoded_coords = self.coord_encoder(coords)

        logits = self.classifier(torch.cat((encoded_coords, feats), dim=-1))
        logits = logits.reshape(*shape, -1)
        return logits


if __name__ == "__main__":

    batch_size = 4

    dset = loader.Dataset()
    trainloader = torch.utils.data.DataLoader(
        dset,
        num_workers=10,
        drop_last=True,
        pin_memory=True,
        shuffle=False,
        batch_size=batch_size,
        worker_init_fn=lambda worker_id: np.random.seed(),
    )

    input_height, input_width = fpn.transform(PIL.Image.fromarray(dset[0][0])).shape[1:]

    model = torch.nn.ModuleDict(
        {"cnn": fpn.FPN(input_height, input_width, 1), "mlp": MLP(),}
    ).cuda()

    if config.wandb:
        wandb.init(project="posetest")
        wandb.watch(model)

    # opt = torch.optim.Adam(model.parameters())
    opt = torch.optim.Adam(
        [
            {"params": model["cnn"].parameters(), "lr": 1e-4},
            {"params": model["mlp"].parameters(), "lr": 1e-3},
        ]
    )

    if True:
        checkpoint = torch.load("models/2500-baseline")
        model.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["opt"])

    bce = torch.nn.BCEWithLogitsLoss()

    step = 0
    for epoch in range(1000):
        print("epoch {}".format(epoch))

        model.train()
        for batch in tqdm.tqdm(trainloader):
            (
                rgb_img,
                rgb_img_t,
                depth_img,
                anchor_uv,
                anchor_xyz_cam,
                anchor_xyz_cam_rotated,
                query_xyz_cam,
                query_xyz_cam_rotated,
                query_coords,
                query_occ,
                query_sd,
                query_inds,
                camera_pose,
                cam2anchor_rot,
                index,
            ) = batch
            assert rgb_img.shape[0] == batch_size

            rgb_img_t = rgb_img_t.cuda()
            query_coords = query_coords.cuda()
            # query_occ = query_occ.cuda().float()
            query_sd = query_sd.cuda().float()

            img_feats = model["cnn"](rgb_img_t)
            img_feats = torch.nn.functional.interpolate(
                img_feats,
                size=(rgb_img_t.shape[2:]),
                mode="bilinear",
                align_corners=False,
            )
            anchor_uv_t = (
                (
                    anchor_uv
                    / torch.Tensor([rgb_img.shape[2], rgb_img.shape[1]])
                    * torch.Tensor([img_feats.shape[3], img_feats.shape[2]])
                )
                .float()
                .cuda()
            )

            pixel_feats = []
            for i in range(len(img_feats)):
                pixel_feats.append(interp_img(img_feats[i], anchor_uv_t[i]).T)
            pixel_feats = torch.stack(pixel_feats, dim=0)
            pixel_feats = pixel_feats[:, :, None].repeat(
                (1, 1, query_coords.shape[2], 1)
            )

            logits = model["mlp"](query_coords, pixel_feats)[..., 0]

            preds = torch.tanh(logits)

            """
            j = 0
            subplot(221)
            imshow(rgb_img[j])
            plot(anchor_uv[j, 0], anchor_uv[j, 1], '.')
            subplot(222)
            imshow(depth_img[j])
            plot(anchor_uv[j, 0], anchor_uv[j, 1], '.')
            gcf().add_subplot(223, projection='3d')
            q = query_coords.numpy()
            qc = query_occ.numpy()
            plot(q[j, qc[j], 0], q[j, qc[j], 1], q[j, qc[j], 2], 'b.')
            plot(q[j, ~qc[j], 0], q[j, ~qc[j], 1], q[j, ~qc[j], 2], 'r.')
            plot([0], [0], [0], '.')
            plot([-anchor_xyz_cam[j, 0]], [-anchor_xyz_cam[j, 1]], [-anchor_xyz_cam[j, 2]], '.')
            xlabel('x')
            ylabel('y')

            k = 0

            query_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(query_xyz_cam[j, k]))
            query_pcd.colors = o3d.utility.Vector3dVector(plt.cm.jet(query_sd[j, k] / (2 * dset.query_radius) + .5)[:, :3])
            o3d.visualization.draw_geometries([query_pcd])

            """

            rgb_img_t = rgb_img_t.cuda()
            query_coords = query_coords.cuda()
            # query_occ = query_occ.cuda().float()
            query_sd = query_sd.cuda().float()

            opt.zero_grad()

            img_feats = model["cnn"](rgb_img_t)

            img_feats = torch.nn.functional.interpolate(
                img_feats,
                size=(rgb_img_t.shape[2:]),
                mode="bilinear",
                align_corners=False,
            )

            anchor_uv_t = (
                (
                    anchor_uv
                    / torch.Tensor([rgb_img.shape[2], rgb_img.shape[1]])
                    * torch.Tensor([img_feats.shape[3], img_feats.shape[2]])
                )
                .float()
                .cuda()
            )

            pixel_feats = []
            for i in range(len(img_feats)):
                pixel_feats.append(interp_img(img_feats[i], anchor_uv_t[i]).T)
            pixel_feats = torch.stack(pixel_feats, dim=0)
            pixel_feats = pixel_feats[:, :, None].repeat(
                (1, 1, query_coords.shape[2], 1)
            )

            logits = model["mlp"](query_coords, pixel_feats)[..., 0]

            preds = torch.tanh(logits)

            # plain l1
            target = query_sd / dset.query_radius
            inputs = preds

            # log-transformed l1
            # target = log_transform(query_sd / dset.query_radius)
            # inputs = log_transform(preds)

            loss = torch.abs(target - inputs)
            inds = (torch.abs(query_sd) < 0.01).float()
            loss = torch.sum(loss) / torch.sum(inds)

            # loss = bce(logits, query_occ)
            loss.backward()
            opt.step()

            # preds = torch.sigmoid(logits)

            # pos_inds = query_occ.bool()
            pos_inds = query_sd > 0
            neg_inds = ~pos_inds

            true_pos = torch.sum((preds > 0) & pos_inds).float()
            all_predicted_pos = torch.sum(preds > 0)
            all_actual_pos = torch.sum(pos_inds)
            precision = true_pos / all_predicted_pos
            recall = true_pos / all_actual_pos

            if config.wandb:
                wandb.log(
                    {
                        "loss": loss.item(),
                        # "pos_loss": pos_loss.item(),
                        # "neg_loss": neg_loss.item(),
                        "precision": precision.item(),
                        "recall": recall.item(),
                        # "logits": wandb.Histogram(logits.detach().cpu().numpy()),
                        # "preds": wandb.Histogram(preds.detach().cpu().numpy()),
                        # "occ": wandb.Histogram(query_occ.cpu().numpy()),
                    },
                    step=step,
                )

            step += 1

        if config.wandb:
            """
            eval loop
            """
            model.eval()
            n_correct = torch.zeros(len(dset.query_pts)).cuda()
            n_total = torch.zeros(len(dset.query_pts)).cuda()
            it = iter(tqdm.tqdm(trainloader))
            while torch.mean((n_total >= 10).float()) < 0.1:
                batch = next(it)
                (
                    rgb_img,
                    rgb_img_t,
                    depth_img,
                    anchor_uv,
                    anchor_xyz_cam,
                    anchor_xyz_cam_rotated,
                    query_xyz_cam,
                    query_xyz_cam_rotated,
                    query_coords,
                    query_occ,
                    query_sd,
                    query_inds,
                    camera_pose,
                    cam2anchor_rot,
                    index,
                ) = batch
                rgb_img_t = rgb_img_t.cuda()
                query_coords = query_coords.cuda()
                # query_occ = query_occ.cuda().float()
                query_sd = query_sd.cuda().float()

                img_feats = model["cnn"](rgb_img_t)
                img_feats = torch.nn.functional.interpolate(
                    img_feats,
                    size=(rgb_img_t.shape[2:]),
                    mode="bilinear",
                    align_corners=False,
                )
                anchor_uv_t = (
                    (
                        anchor_uv
                        / torch.Tensor([rgb_img.shape[2], rgb_img.shape[1]])
                        * torch.Tensor([img_feats.shape[3], img_feats.shape[2]])
                    )
                    .float()
                    .cuda()
                )

                pixel_feats = []
                for i in range(len(img_feats)):
                    pixel_feats.append(interp_img(img_feats[i], anchor_uv_t[i]).T)
                pixel_feats = torch.stack(pixel_feats, dim=0)
                pixel_feats = pixel_feats[:, :, None].repeat(
                    (1, 1, query_coords.shape[2], 1)
                )

                logits = model["mlp"](query_coords, pixel_feats)[..., 0]
                preds = torch.tanh(logits)
                correct = ((preds > 0) == (query_sd > 0)).float()
                for i in range(correct.shape[0]):
                    for j in range(correct.shape[1]):
                        n_correct[query_inds[i, j]] += correct[i, j]
                        n_total[query_inds[i, j]] += 1

            acc = (n_correct / n_total).cpu().numpy()
            inds = (n_total.cpu().numpy() >= 10) & (acc < 0.98)

            plot3()
            plot(dset.query_pts[:, 0], dset.query_pts[:, 1], dset.query_pts[:, 2], ".")
            xlabel("x")
            ylabel("y")

            plt.figure()
            plt.subplot(131)
            plt.scatter(
                dset.query_pts[inds, 0],
                dset.query_pts[inds, 2],
                c=plt.cm.jet(acc[inds]),
                alpha=0.7,
            )
            plt.axis("off")
            plt.subplot(132)
            plt.scatter(
                dset.query_pts[inds, 0],
                dset.query_pts[inds, 1],
                c=plt.cm.jet(acc[inds]),
                alpha=0.7,
            )
            plt.axis("off")
            plt.subplot(133)
            plt.scatter(
                dset.query_pts[inds, 1],
                dset.query_pts[inds, 2],
                c=plt.cm.jet(acc[inds]),
                alpha=0.7,
            )
            plt.axis("off")
            plt.tight_layout()
            plt.gcf().set_size_inches(16, 8)
            plt.savefig("plots.png", bbox_inches="tight", dpi=300)
            plt.close()

            wandb.log({"test plots": wandb.Image("plots.png")}, step=step)

        torch.save(
            {"model": model.state_dict(), "opt": opt.state_dict()},
            os.path.join("models", "test-{}".format(step)),
        )
