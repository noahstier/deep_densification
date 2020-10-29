import datetime
import glob
import itertools
import os

import numpy as np
import open3d as o3d
import pytorch_lightning as pl
import skimage.measure
import torch
import tqdm

import common
import config
import loader
import decoders
import pointnet
import pointnetpp

"""
todo:

    train overnight

    ResnetPointnet
        - uses FC layers -- can switch to Conv
        - uses batch norm. try turning off?

    loss
        - incorporate some uniformly spread query points, with loss scaled by TSDF?
            - maybe don't need to scale loss
        - try: visualize per-query point accuracy + loss for trained model


to improve:
    more scans
    cbatchnorm?
    image feats
    pointnet++

unstable test performance?
    - tune batch norm momentum. try .1 or .5
    - normalization: divide by std, divide by 5, normalize to unit sphere, no norm
"""


import onet.encoder
import onet.decoder


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = onet.encoder.ResnetPointnet(c_dim=1024, dim=6, hidden_dim=128)
        self.decoder = onet.decoder.DecoderBatchNorm(
            dim=3, z_dim=0, c_dim=1024, hidden_size=256
        )

    def forward(self, input_xyz, rgb, query_xyz):
        pointnet_inputs = torch.cat((input_xyz / 3, rgb * 2 - 1), dim=-1)
        x = self.encoder(pointnet_inputs)
        x = self.decoder(query_xyz, None, x)
        return x

    def training_step(self, batch, batch_idx):
        # pts, rgb, query_coords, query_tsdf, rgb_imgs, pt_inds = batch
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
        near_inds = query_tsdf < 1

        logits = self(pts, rgb, query_coords)

        logits = logits[near_inds]
        query_occ = query_occ[near_inds]

        # loss = torch.nn.functional.binary_cross_entropy_with_logits(
        #     logits, query_occ.float()
        # )

        acc = ((logits > 0) == query_occ).float().mean()

        pos_logits = logits[query_occ]
        neg_logits = logits[~query_occ]

        pos_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pos_logits, torch.ones_like(pos_logits)
        )
        neg_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            neg_logits, torch.zeros_like(neg_logits)
        )

        loss = (pos_loss + neg_loss) / 2

        self.log_dict(
            {
                "train/loss": loss,
                "train/acc": acc,
                "train/pos_loss": pos_loss,
                "train/neg_loss": neg_loss,
            },
            on_epoch=True,
            on_step=False,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        pts, rgb, query_coords, query_tsdf, = batch
        # pts, rgb, query_coords, query_tsdf, rgb_imgs, pt_inds = batch

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
        near_inds = query_tsdf < 1

        logits = self(pts, rgb, query_coords)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[near_inds], query_occ[near_inds].float()
        )

        acc = ((logits[near_inds] > 0) == query_occ[near_inds]).float().mean()

        self.log_dict(
            {"test/loss": loss, "test/acc": acc,}, on_epoch=True, on_step=False,
        )

    def validation_epoch_end(self, outputs):
        # pts, rgb, _, _, _, _ = test_batch
        pts, rgb, _, _ = test_batch
        try:
            verts, faces, preds, query_coords = common.predict_mesh(model, pts, rgb)
            test_mesh_img = common.render_mesh(verts, faces)
            mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces)
            )
            mesh.compute_vertex_normals()
            mesh_dir = os.path.join("meshes", self.logger.name)
            os.makedirs(mesh_dir, exist_ok=True)
            o3d.io.write_triangle_mesh(
                os.path.join(mesh_dir, str(self.global_step).zfill(7) + ".ply"), mesh
            )
            self.logger.experiment.add_image(
                "test/mesh",
                test_mesh_img,
                dataformats="HWC",
                global_step=self.global_step,
            )
        except Exception as e:
            print("meshing failed", e)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


if __name__ == "__main__":

    scan_dirs = common.get_scan_dirs(
        config.scannet_dir,
        scene_types=[
            "Living room / Lounge",
            "Bathroom",
            "Bedroom / Hotel",
            "Kitchen",
            "Conference Room",
            "Copy/Mail Room",
            "Office",
        ],
    )
    scan_dirs = [d for d in scan_dirs if d.endswith("00")]

    # train_dset = loader.CachedDataset("batches", augment=True)
    train_dset = loader.Dataset(
        scan_dirs[2:],
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
        pin_memory=True
    )

    '''
    batch = next(iter(train_loader))
    (pts, rgb, query_coords, query_tsdf, rgb_imgs, pt_inds) = batch

    import unet
    cnn = unet.UNet(3, None)
    r = rgb_imgs.transpose(4, 3).transpose(3, 2)
    r = r.reshape(-1, *r.shape[2:])
    feats = cnn(r)
    feats = feats.reshape(config.batch_size, -1, *feats.shape[1:])

    j = 1
    feats[j, pt_inds[j, :, 0], 
    '''

    test_dset = loader.Dataset(
        scan_dirs[:2],
        n_imgs=17,
        augment=False,
        maxqueries=config.maxqueries,
        maxpts=config.maxpts,
    )
    # test_dset = loader.CachedDataset("batches")
    test_batch = test_dset[0]
    test_batch = [torch.Tensor(i).cuda() for i in test_batch]
    test_loader = torch.utils.data.DataLoader(
        test_dset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        worker_init_fn=lambda worker_id: np.random.seed(),
        pin_memory=True
    )

    model = Model()
    checkpoint_path = None
    # checkpoint_path = "logs/high-level-line/version_0/checkpoints/epoch=38.ckpt"

    logger = pl.loggers.TensorBoardLogger("logs", name=common.get_run_name())
    print("RUN NAME: " + logger.name)

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=int(1e10),
        logger=logger,
        resume_from_checkpoint=checkpoint_path,
        check_val_every_n_epoch=1,
    )
    trainer.fit(model, train_loader, test_loader)
