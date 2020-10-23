import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops.pointnet2_modules import PointnetSAModule, PointnetSAModuleMSG

from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG


class PointNetPPMSG(PointNet2ClassificationSSG):
    def _build_model(self):
        super()._build_model()

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=512,
                radii=[0.1, 0.2, 0.4],
                nsamples=[16, 32, 128],
                mlps=[[3, 32, 32, 64], [3, 64, 64, 128], [3, 64, 96, 128]],
                use_xyz=self.hparams["model.use_xyz"],
            )
        )

        input_channels = 64 + 128 + 128
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=[0.2, 0.4, 0.8],
                nsamples=[32, 64, 128],
                mlps=[
                    [input_channels, 64, 64, 128],
                    [input_channels, 128, 128, 256],
                    [input_channels, 128, 128, 256],
                ],
                use_xyz=self.hparams["model.use_xyz"],
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                # mlp=[128 + 256 + 256, 256, 512, 1024],
                mlp=[128 + 256 + 256, 256],
                use_xyz=self.hparams["model.use_xyz"],
            )
        )

        self.var = torch.nn.Parameter(
            torch.Tensor([1.37907848, 2.33269393, 0.40424237]), requires_grad=False
        )

    def forward(self, pointcloud):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)
        xyz = xyz / self.var

        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        return features.squeeze(-1)


class PointNetPPSSG(PointNet2ClassificationSSG):
    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=512,
                # radius=0.2,
                radius=1,
                nsample=64,
                mlp=[3, 64, 64, 128],
                use_xyz=self.hparams["model.use_xyz"],
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                # radius=0.4,
                radius=2,
                nsample=64,
                mlp=[128, 128, 128, 256],
                use_xyz=self.hparams["model.use_xyz"],
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[256, 256, 512],
                use_xyz=self.hparams["model.use_xyz"]
                # mlp=[256, 256, 512, 1024], use_xyz=self.hparams["model.use_xyz"]
            )
        )

        self.std = torch.nn.Parameter(
            torch.Tensor([1.37907848, 2.33269393, 0.40424237]), requires_grad=False
        )

    def forward(self, pointcloud):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)
        xyz = xyz / self.std

        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        return features.squeeze(-1)
