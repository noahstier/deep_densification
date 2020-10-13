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


def focal_loss(inputs, targets, gamma):
    pt = (inputs * targets) + (1 - inputs) * (1 - targets)
    loss = -((1 - pt) ** gamma) * torch.log(pt)
    return loss


def bn_relu_fc(in_c, out_c):
    return torch.nn.Sequential(
        torch.nn.BatchNorm1d(in_c),
        torch.nn.ReLU(),
        torch.nn.Linear(in_c, out_c, bias=False),
    )


class Pointnet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(3, 32, bias=False),
            bn_relu_fc(32, 64),
            bn_relu_fc(64, 128),
            bn_relu_fc(128, 256),
            bn_relu_fc(256, 256),
        )

    def forward(self, pts):
        shape = pts.shape[:-1]
        pts = pts.reshape(np.prod(list(shape)), pts.shape[-1])
        return self.mlp(pts).reshape(*shape, -1)


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
            bn_relu_fc(512, 512),
            bn_relu_fc(512, 512),
            bn_relu_fc(512, 512),
            bn_relu_fc(512, 512),
            bn_relu_fc(512, 512),
            bn_relu_fc(512, 128),
            bn_relu_fc(128, 32),
            bn_relu_fc(32, 1),
        )

    def forward(self, coords, feats):
        shape = coords.shape[:-1]
        coords = coords.reshape(np.prod(list(shape)), coords.shape[-1])
        feats = feats.reshape(np.prod(list(shape)), feats.shape[-1])

        encoded_coords = self.coord_encoder(coords)

        logits = self.classifier(torch.cat((encoded_coords, feats), dim=-1))
        logits = logits.reshape(*shape, -1)
        return logits


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


if __name__ == "__main__":
    torch.manual_seed(0)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(0)

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

    # included_classes = [5, 11, 20, 67, 72]
    included_classes = [11]
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
        num_workers=6,
        drop_last=True,
        pin_memory=True,
    )

    input_height, input_width = fpn.transform(PIL.Image.fromarray(dset[0][-4])).shape[
        1:
    ]

    model = torch.nn.ModuleDict(
        {
            "cnn": fpn.FPN(input_height, input_width, 1),
            "mlp": MLP(),
            "pointnet": Pointnet(),
        }
    ).cuda()

    optimizer = torch.optim.Adam(
        [
            {"params": model["cnn"].parameters(), "lr": 1e-4},
            {"params": model["mlp"].parameters(), "lr": 1e-3},
            {"params": model["pointnet"].parameters(), "lr": 1e-3},
        ]
    )

    if False:
        checkpoint = torch.load("models/5-class-128pt")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["opt"])

    if config.wandb:
        wandb.init(project="deepmvs")
        wandb.watch(model)

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
                near_sfm_coords,
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

            rgb_img_t = rgb_img_t.cuda()
            query_coords = query_coords.cuda()
            query_tsdf = query_tsdf.float().cuda()
            query_uv_t = query_uv_t.cuda()
            query_occ = (query_tsdf <= 0).float()
            near_sfm_coords = near_sfm_coords.float().cuda()

            optimizer.zero_grad()

            # pointnet_feats = torch.max(model['pointnet'](near_sfm_coords), dim=2)[0]

            img_feats = model["cnn"](rgb_img_t)

            anchor_uv_t_t = (
                anchor_uv_t
                / torch.Tensor([rgb_img_t.shape[3], rgb_img_t.shape[2]])
                * torch.Tensor([img_feats.shape[3], img_feats.shape[2]])
            ).cuda()

            pixel_feats = []
            for i in range(len(img_feats)):
                pixel_feats.append(interp_img(img_feats[i], anchor_uv_t_t[i]).T)
            pixel_feats = torch.stack(pixel_feats, dim=0).float()

            # anchor_feats = torch.cat((pixel_feats, regional_feats, neighborhood_feats), dim=-1)
            anchor_feats = pixel_feats
            anchor_feats = anchor_feats[:, :, None].repeat(
                (1, 1, query_coords.shape[2], 1)
            )

            logits = model["mlp"](query_coords, anchor_feats)[..., 0]

            preds = torch.sigmoid(logits)

            loss = occ_loss_fn(preds, query_occ)
            # loss = focal_loss(preds, query_occ, config.gamma)

            loss = torch.mean(loss)

            loss.backward()
            optimizer.step()

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

            # pos_loss = torch.mean(occ_loss_fn(preds[pos_inds], query_occ[pos_inds]))
            # neg_loss = torch.mean(occ_loss_fn(preds[neg_inds], query_occ[neg_inds]))

            # loss = pos_loss + neg_loss

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

            step += 1
            if config.wandb:
                wandb.log(
                    {
                        # "pos loss": pos_loss.item(),
                        # "neg loss": neg_loss.item(),
                        "loss": loss.item(),
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

        name = "all-classes-{}".format(step)
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
