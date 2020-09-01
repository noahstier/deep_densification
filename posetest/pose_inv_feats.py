_wandb = True

import glob
import os

import h5py
import PIL.Image
import numpy as np
import open3d as o3d
import scipy.spatial
import torch
import torchvision
import tqdm

import wandb

torch.autograd.set_detect_anomaly(True)


if _wandb:
    wandb.init(project="posetest")

torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(1)

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


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


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.dset = h5py.File("fuze.hdf5", "r")
        self.rgb_dset = self.dset["rgb_imgs"]
        self.depth_dset = self.dset["depth_imgs"]
        self.pose_dset = self.dset["pose"]

    def __len__(self):
        return len(self.rgb_dset)

    def __getitem__(self, index):
        rgb_img = self.rgb_dset[index]
        pil_img = PIL.Image.fromarray(rgb_img)
        rgb_img_t = transform(pil_img)
        pose = self.pose_dset[index]
        if np.any(np.isnan(pose)):
            return self[np.random.randint(len(self))]

        return rgb_img, rgb_img_t, pose


batch_size = 32

dset = Dataset()
loader = torch.utils.data.DataLoader(
    dset,
    num_workers=8,
    drop_last=True,
    pin_memory=True,
    shuffle=True,
    batch_size=batch_size,
)

cnn = torchvision.models.mobilenet_v2(pretrained=False).features
mlp_up = torch.nn.Sequential(
    FCLayer(1, 64), FCLayer(64), FCLayer(64), torch.nn.Linear(64, 3)
)
mlp_down = torch.nn.Sequential(
    FCLayer(3, 64), FCLayer(64), FCLayer(64), torch.nn.Linear(64, 1)
)
model = torch.nn.ModuleList([cnn, mlp_up, mlp_down]).cuda()

if _wandb:
    wandb.watch(model)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
tri_inds = torch.triu(torch.ones((batch_size, batch_size)), diagonal=1).bool().cuda()

cs = torch.nn.CosineSimilarity(dim=-1)
for batch in tqdm.tqdm(loader):
    opt.zero_grad()

    rgb_img, rgb_img_t, pose = batch
    rgb_img_t = rgb_img_t.cuda()
    pose = pose.cuda()

    feats = cnn(rgb_img_t)[:, :400].mean([2, 3])[..., None]
    feats3d = mlp_up(feats)
    feats3d = torch.softmax(feats3d, dim=-1)
    norms = torch.norm(feats3d, dim=-1, keepdim=True)
    zero_inds = (norms == 0).float()
    norms = torch.ones_like(norms).cuda() * zero_inds + (1 - zero_inds) * norms
    feats3d_u = feats3d / norms

    feats3d_t = (pose[:, :3, :3].inverse() @ feats3d_u.transpose(1, 2)).transpose(1, 2)
    angles = torch.acos(
        torch.clamp(
            torch.sum(feats3d_t[:, None] - feats3d_t[None], dim=-1), 0.001, 0.999
        )
    )
    loss = torch.mean(torch.mean(angles, dim=-1)[tri_inds])
    loss.backward()

    opt.step()

    if _wandb:
        wandb.log(
            {
                "loss": loss,
                "feats hist": wandb.Histogram(feats.detach().cpu().numpy().flatten()),
            }
        )


"""

logits = torch.randn(4, 5)
onehot_target = torch.Tensor([
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
])
target = torch.argmax(onehot_target, dim=-1)
l = torch.nn.CrossEntropyLoss()
print(l(a, b))

preds = torch.exp(logits) / torch.sum(torch.exp(logits), dim=-1, keepdims=True)
-torch.sum(onehot_target * torch.log(preds))
"""
