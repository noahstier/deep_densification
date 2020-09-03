import glob
import logging
import os

import cv2
import h5py
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


class res_block(torch.nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(n_channels, n_channels, bias=False),
            torch.nn.BatchNorm1d(n_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(n_channels, n_channels, bias=False),
            torch.nn.BatchNorm1d(n_channels),
        )

    def forward(self, inputs):
        return torch.nn.functional.relu(self.layers(inputs) + inputs)


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.coord_encoder = torch.nn.Sequential(
            torch.nn.Linear(3, 32, bias=False),
            bn_relu_fc(32, 32),
            bn_relu_fc(32, 64),
            bn_relu_fc(64, 128),
            bn_relu_fc(128, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
        )

        self.offsetter = torch.nn.Sequential(
            torch.nn.Linear(896, 512, bias=False),
            bn_relu_fc(512, 256),
            bn_relu_fc(256, 256),
            bn_relu_fc(256, 256),
            bn_relu_fc(256, 256),
            bn_relu_fc(256, 256),
            bn_relu_fc(256, 256),
        )

        self.classifier = torch.nn.Sequential(
            bn_relu_fc(256, 64),
            bn_relu_fc(64, 16),
            bn_relu_fc(16, 1),
        )

    def forward(self, coords, feats):
        shape = coords.shape[:-1]
        coords = coords.reshape(np.prod(list(shape)), coords.shape[-1])
        feats = feats.reshape(np.prod(list(shape)), feats.shape[-1])

        encoded_coords = self.coord_encoder(coords)

        offset = self.offsetter(torch.cat((encoded_coords, feats), dim=-1))
        logits = self.classifier(offset)
        logits = logits.reshape(*shape, -1)
        return logits


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.dsetdir = "dset-quant-42"

        self.rgb_imgfiles = sorted(glob.glob(os.path.join(self.dsetdir, "rgb/*.jpg")))
        self.depth_imgfiles = sorted(
            glob.glob(os.path.join(self.dsetdir, "depth/*.png"))
        )
        self.poses = np.load(os.path.join(self.dsetdir, "poses.npy"))
        self.intrinsic = np.load(os.path.join(self.dsetdir, "intrinsic.npy"))
        self.inv_intrinsic = np.linalg.inv(self.intrinsic)

        self.rotinds = np.load(os.path.join(self.dsetdir, "rotinds.npy"))

        query_npz = np.load(os.path.join(self.dsetdir, "sdf.npz"))
        self.query_pts = query_npz["pts"]
        self.query_sdf = query_npz["sd"]

        self.poses = np.load(os.path.join(self.dsetdir, "poses.npy"))

        self.mesh = trimesh.load("fuze.obj")
        v = np.asarray(self.mesh.vertices)
        vertmean = np.mean(v, axis=0)
        diaglen = np.linalg.norm(np.max(v, axis=0) - np.min(v, axis=0))
        v = (v - vertmean) / diaglen
        self.mesh.vertices = v

        # self.maxnorm = np.linalg.norm(np.max(self.mesh.vertices, axis=0))
        self.n_normal = 200
        self.n_uniform = 256
        self.query_radius = 0.2

        self.n_anchors = 8

    def __len__(self):
        """
        """
        return len(self.rgb_imgfiles)
        """
        """
        # return len(self.rgb_imgfiles)

    def __getitem__(self, index):
        """
        """
        # index = 0
        """
        """

        depth_img = (
            cv2.imread(self.depth_imgfiles[index], cv2.IMREAD_ANYDEPTH).astype(
                np.float32
            )
            / 1000
        )
        pil_img = PIL.Image.open(self.rgb_imgfiles[index])
        rgb_img = np.asarray(pil_img).copy()
        rgb_img_t = fpn.transform(pil_img)

        # rgb_img = rgb_img - ((rgb_img == 255) * np.random.randint(100, size=rgb_img.shape)).astype(np.uint8)
        pose = self.poses[index]

        uv = np.argwhere(depth_img > 0)[:, [1, 0]]
        inds = np.arange(len(uv))
        anchor_inds = uv[np.random.choice(inds, size=self.n_anchors, replace=False)]

        """
        todo
        """
        anchor_uv = anchor_inds + 0.5
        anchor_range = depth_img[anchor_inds[:, 1], anchor_inds[:, 0]]
        anchor_xyz_cam = (
            (self.inv_intrinsic @ np.c_[anchor_uv, np.ones(len(anchor_uv))].T)
            * -anchor_range
        ).T

        anchor_xyz_canonical = (
            np.linalg.inv(pose) @ np.c_[anchor_xyz_cam, np.ones(len(anchor_xyz_cam))].T
        ).T[:, :3]

        in_range_inds = (
            np.linalg.norm(anchor_xyz_canonical[:, None] - self.query_pts, axis=-1)
            < self.query_radius
        )

        query_tsdf = []
        query_xyz_cam = []
        for i in range(self.n_anchors):
            inds = np.random.choice(
                np.argwhere(in_range_inds[i]).flatten(),
                replace=False,
                size=self.n_uniform,
            )
            query_pts = self.query_pts[inds]
            query_tsdf.append(self.query_sdf[inds])
            query_xyz_cam.append(
                (pose @ np.c_[query_pts, np.ones(len(query_pts))].T).T[:, :3]
            )

        query_xyz_cam = np.stack(query_xyz_cam, axis=0)
        query_sd = np.stack(query_tsdf, axis=0)

        # x = np.random.normal(0, 1, size=(self.n_uniform, 5))
        # uniform_query_pts = (
        #     x[:, :3] / np.linalg.norm(x, axis=-1, keepdims=True) * self.maxnorm * 1.1
        # )
        # normal_query_pts = self.mesh.sample(self.n_normal) + np.random.normal(
        #     0, 0.05, size=(self.n_normal, 3)
        # )
        # query_pts = np.concatenate((uniform_query_pts, normal_query_pts), axis=0)

        # x = np.random.normal(0, 1, size=(self.n_anchors, self.n_uniform, 5))
        # query_xyz_cam = (
        #     x[..., :3] / np.linalg.norm(x, axis=-1, keepdims=True) * self.query_radius
        #     + anchor_xyz_cam[:, None]
        # )

        # query_sd = np.stack(
        #     [self.mesh.nearest.signed_distance(query_pts[i]) for i in range(self.n_anchors)], axis=0
        # )
        query_occ = query_sd > 0
        # query_xyz_cam = (pose @ np.c_[query_pts, np.ones(len(query_pts))].T).T[:, :3]

        anchor_cam_unit = anchor_xyz_cam / np.linalg.norm(
            anchor_xyz_cam, axis=-1, keepdims=True
        )
        center_pixel_u = np.array([0, 0, -1])

        xz_plane_projection = np.array([1, 0, 1]) * anchor_cam_unit
        xz_plane_projection /= np.linalg.norm(
            xz_plane_projection, axis=-1, keepdims=True
        )
        rot_axis = np.cross(center_pixel_u, xz_plane_projection)
        rot_axis /= np.linalg.norm(rot_axis, axis=-1, keepdims=True)
        horiz_angle = np.arccos(np.dot(xz_plane_projection, center_pixel_u))
        horiz_rot = scipy.spatial.transform.Rotation.from_rotvec(
            rot_axis * horiz_angle[:, None]
        ).as_matrix()

        yz_plane_projection = np.array([0, 1, 1]) * (
            (np.linalg.inv(horiz_rot) @ anchor_cam_unit.T)[
                np.arange(self.n_anchors), :, np.arange(self.n_anchors)
            ]
        )
        yz_plane_projection /= np.linalg.norm(
            yz_plane_projection, axis=-1, keepdims=True
        )
        rot_axis = np.cross(center_pixel_u, yz_plane_projection)
        rot_axis /= np.linalg.norm(rot_axis, axis=-1, keepdims=True)
        vert_angle = np.arccos(np.dot(yz_plane_projection, center_pixel_u))
        vert_rot = scipy.spatial.transform.Rotation.from_rotvec(
            rot_axis * vert_angle[:, None]
        ).as_matrix()

        cam2anchor_rot = horiz_rot @ vert_rot

        """
        todo
        """

        """
        cross = np.cross(center_pixel_u, anchor_cam_unit)
        cross /= np.linalg.norm(cross, axis=-1, keepdims=True)
        dot = np.dot(center_pixel_u, anchor_cam_unit.T)
        axis = cross
        if np.any(np.isnan(axis)):
            return self[np.random.randint(0, len(self))]
        angle = np.arccos(dot) 
        cam2anchor_rot = scipy.spatial.transform.Rotation.from_rotvec(axis * angle[:, None]).as_matrix()
        """

        query_xyz_cam_rotated = np.stack(
            [
                (np.linalg.inv(cam2anchor_rot[i]) @ query_xyz_cam[i].T).T
                for i in range(self.n_anchors)
            ],
            axis=0,
        )
        anchor_xyz_cam_rotated = (np.linalg.inv(cam2anchor_rot) @ anchor_xyz_cam.T)[
            np.arange(self.n_anchors), :, np.arange(self.n_anchors)
        ]

        query_coords = (
            query_xyz_cam_rotated - anchor_xyz_cam_rotated[:, None]
        ) / self.query_radius
        # query_coords = positional_encoding(query_coords, L=2)

        """

        verts = (pose @ np.c_[self.mesh.vertices, np.ones(len(self.mesh.vertices))].T).T[:, :3]
        uv = np.argwhere(depth_img > 0)[:, [1, 0]]
        ranges = depth_img[uv[:, 1], uv[:, 0]]
        xyz_cam = (self.inv_intrinsic @ np.c_[uv, np.ones(len(uv))].T).T * -ranges[:, None]

        gcf().add_subplot(111, projection='3d')
        plot(verts[:, 0], verts[:, 1], verts[:, 2], '.')
        plot(xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2], '.')

        figure()
        subplot(221)
        imshow(rgb_img)
        plot(anchor_uv[0], anchor_uv[1], '.')
        gcf().add_subplot(222, projection='3d')
        plot(query_xyz_cam[query_occ, 0], query_xyz_cam[query_occ, 1], query_xyz_cam[query_occ, 2], '.')
        plot(query_xyz_cam[~query_occ, 0], query_xyz_cam[~query_occ, 1], query_xyz_cam[~query_occ, 2], '.')
        plot([anchor_xyz_cam[0]], [anchor_xyz_cam[1]], [anchor_xyz_cam[2]], '.')
        plot([0], [0], [0], '.')
        plot(verts[:, 0], verts[:, 1], verts[:, 2], 'k.', markersize=.1)
        plot(xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2], '.')
        xlabel('x')
        ylabel('y')
        gcf().add_subplot(223, projection='3d')
        plot(query_xyz_cam_rotated[query_occ, 0], query_xyz_cam_rotated[query_occ, 1], query_xyz_cam_rotated[query_occ, 2], '.')
        plot(query_xyz_cam_rotated[~query_occ, 0], query_xyz_cam_rotated[~query_occ, 1], query_xyz_cam_rotated[~query_occ, 2], '.')
        plot([anchor_xyz_cam_rotated[0]], [anchor_xyz_cam_rotated[1]], [anchor_xyz_cam_rotated[2]], '.')
        plot([0], [0], [0], '.')
        xlabel('x')
        ylabel('y')
        gcf().add_subplot(224, projection='3d')
        plot(query_coords[query_occ, 0], query_coords[query_occ, 1], query_coords[query_occ, 2], '.')
        plot(query_coords[~query_occ, 0], query_coords[~query_occ, 1], query_coords[~query_occ, 2], '.')
        plot([0], [0], [0], '.')
        xlabel('x')
        ylabel('y')
        """

        return (
            rgb_img,
            rgb_img_t,
            depth_img,
            anchor_uv,
            anchor_xyz_cam,
            anchor_xyz_cam_rotated,
            query_xyz_cam,
            query_xyz_cam_rotated,
            query_coords.astype(np.float32),
            query_occ,
            query_sd,
            pose,
            cam2anchor_rot,
            index,
            self.rotinds[index],
        )


if __name__ == "__main__":

    batch_size = 16

    dset = Dataset()
    """
    inds = np.arange(len(dset))
    np.random.shuffle(inds)
    for i in tqdm.tqdm(inds):
        batch = dset[i]
        rgb_img, rgb_img_t, anchor_uv, query_xyz_cam, query_occ = batch
        if np.any(np.isnan(query_xyz_cam)):
            break
    """

    loader = torch.utils.data.DataLoader(
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
        {
            "cnn": fpn.FPN(input_height, input_width, 1),
            "mlp": MLP(),
            "rot_encoder": torch.nn.Sequential(
                bn_relu_fc(42, 64),
                bn_relu_fc(64, 128),
                bn_relu_fc(128, 128),
                bn_relu_fc(128, 128),
                bn_relu_fc(128, 128),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU()
            ),
            # "rot_class": torch.nn.Sequential(
            #     bn_relu_fc(128, 128),
            #     bn_relu_fc(128, 128),
            #     bn_relu_fc(128, 128),
            #     bn_relu_fc(128, 64),
            #     bn_relu_fc(64, 42),
            # ),
        }
    ).cuda()

    if False:
        checkpoint = torch.load("models/test")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["opt"])

    if config.wandb:
        wandb.init(project="posetest")
        wandb.watch(model)

    # opt = torch.optim.Adam(model.parameters())
    opt = torch.optim.Adam(
        [
            {"params": model["cnn"].parameters(), "lr": 1e-4},
            {"params": model["mlp"].parameters(), "lr": 1e-3},
            # {"params": model["rot_class"].parameters(), "lr": 1e-3},
            {"params": model["rot_encoder"].parameters(), "lr": 1e-3},
        ]
    )

    bce = torch.nn.BCEWithLogitsLoss()
    ce = torch.nn.CrossEntropyLoss().cuda()

    step = 0
    for epoch in range(1000):
        print("epoch {}".format(epoch))
        model.train()
        for batch in tqdm.tqdm(loader):
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
                pose,
                cam2anchor_rot,
                index,
                rotind,
            ) = batch
            assert rgb_img.shape[0] == batch_size

            # if np.any([torch.any(torch.isnan(a)) for a in batch]):
            #     raise Exception('gah')

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
            rotind = rotind.cuda().long()

            opt.zero_grad()

            rot_feats = model['rot_encoder'](torch.nn.functional.one_hot(rotind, num_classes=42).float())
            rot_feats = rot_feats[:, None, None].repeat(1, dset.n_anchors, dset.n_uniform, 1)


            # if np.any([torch.any(torch.isnan(a)) for a in model['cnn'].parameters()]):
            #     raise Exception('gah')
            img_feats, _, fpn_features = model["cnn"](rgb_img_t)
            # if np.any([torch.any(torch.isnan(a)) for a in model['cnn'].parameters()]):
            #     raise Exception('gah')
            # if np.any([torch.any(torch.isinf(a)) for a in model['cnn'].parameters()]):
            #     raise Exception('gah')
            # if torch.any(torch.isnan(img_feats)):
            #     raise Exception('gah')

            featmeans = np.array(
                [torch.mean(fpn_features[i].detach()).item() for i in range(4)]
            )
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

            # rotind = rotind[:, None].repeat(1, pixel_feats.shape[1]).reshape(-1)
            # rot_logits = model['rot_class'](pixel_feats.reshape(-1, pixel_feats.shape[-1]))
            # rot_loss = ce(rot_logits, rotind)


            pixel_feats = pixel_feats[:, :, None].repeat(
                (1, 1, query_coords.shape[2], 1)
            )
            a = torch.cat((pixel_feats, rot_feats), dim=-1)

            logits = model["mlp"](query_coords, a)[..., 0]

            preds = torch.tanh(logits)

            # plain l1
            target = query_sd / dset.query_radius
            inputs = preds

            # log-transformed l1
            # target = log_transform(query_sd / dset.query_radius)
            # inputs = log_transform(preds)


            sd_loss = torch.abs(target - inputs)
            inds = (torch.abs(query_sd) < 0.02).float()
            sd_loss = torch.sum(sd_loss) / torch.sum(inds)

            loss = sd_loss
            # loss = rot_loss + sd_loss
            # loss = rot_loss

            # rot_acc = torch.mean((torch.argmax(rot_logits, dim=-1) == rotind).float()).item()


            # loss = bce(logits, query_occ)
            loss.backward()
            opt.step()

            # if np.any([torch.any(torch.isnan(a)) for a in model['cnn'].parameters()]):
            #     raise Exception('gah')

            # preds = torch.sigmoid(logits)

            # pos_inds = query_occ.bool()
            pos_inds = query_sd > 0
            neg_inds = ~pos_inds

            true_pos = torch.sum((preds > 0) & pos_inds).float()
            all_predicted_pos = torch.sum(preds > 0)
            all_actual_pos = torch.sum(pos_inds)
            precision = true_pos / all_predicted_pos
            recall = true_pos / all_actual_pos

            # if torch.any(torch.isnan(logits)).item():
            #     raise Exception('gah')

            if config.wandb:
                wandb.log(
                    {
                        "loss": loss.item(),
                        # "rot_acc": rot_acc,
                        # "pos_loss": pos_loss.item(),
                        # "neg_loss": neg_loss.item(),
                        "precision": precision.item(),
                        "recall": recall.item(),
                        **{"featmean {}".format(i): featmeans[i] for i in range(4)}
                        # "logits": wandb.Histogram(logits.detach().cpu().numpy()),
                        # "preds": wandb.Histogram(preds.detach().cpu().numpy()),
                        # "occ": wandb.Histogram(query_occ.cpu().numpy()),
                    },
                    step=step,
                )

            step += 1

        torch.save(
            {"model": model.state_dict(), "opt": opt.state_dict()},
            os.path.join("models", "test-{}".format(step)),
        )
