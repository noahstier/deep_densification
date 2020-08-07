_wandb = False

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

sphere = o3d.geometry.TriangleMesh.create_sphere(resolution=10)
sphere_verts = np.asarray(sphere.vertices)
angles = np.arccos(np.dot(sphere_verts, [1, 0, 0]))
axes = np.cross(sphere_verts, [1, 0, 0])
rotvecs = axes / np.linalg.norm(axes, axis=-1, keepdims=True) * angles[:, None]
gt_rots = scipy.spatial.transform.Rotation.from_rotvec(rotvecs)
gt_quats = gt_rots.as_quat()
gt_euler = gt_rots.as_euler('xyz')
gt_rotmat = gt_rots.as_matrix()

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


def positional_encoding(xyz, L):
    encoding = []
    for l in range(L):
        encoding.append(np.sin(2 ** l ** np.pi * xyz))
        encoding.append(np.cos(2 ** l ** np.pi * xyz))
    encoding = np.concatenate(encoding, axis=-1)
    return encoding


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

        rot = scipy.spatial.transform.Rotation.from_matrix(pose[:3, :3])
        gt_rotmat = rot.as_matrix()
        gt_quat = rot.as_quat()

        if np.any(np.isnan(gt_quat)):
            return self[np.random.randint(0, len(self))]

        gt_dists = 2 * np.arccos(np.abs(np.sum(gt_quats * gt_quat, axis=-1)))
        loss_weights = np.exp(-gt_dists)

        return rgb_img, rgb_img_t, gt_rotmat, gt_quat, loss_weights


batch_size = 16

dset = Dataset()
loader = torch.utils.data.DataLoader(
    dset,
    num_workers=8,
    drop_last=True,
    pin_memory=True,
    shuffle=True,
    batch_size=batch_size,
)

cnn = torchvision.models.mobilenet_v2(pretrained=True).features

predictor_mlp = torch.nn.Sequential(
    FCLayer(1280, 1024),
    FCLayer(1024, 512),
    FCLayer(512),
    FCLayer(512),
    torch.nn.Linear(512, len(gt_quats)),
)

model = torch.nn.ModuleDict({"cnn": cnn, "predictor_mlp": predictor_mlp}).cuda()
if _wandb:
    wandb.watch(model)

opt = torch.optim.Adam(model.parameters())


crossentropy = torch.nn.CrossEntropyLoss(reduction='none')

for batch in tqdm.tqdm(loader):
    opt.zero_grad()

    rgb_img, rgb_img_t, gt_rotmat, gt_quat, loss_weights = batch
    rgb_img_t = rgb_img_t.cuda()
    loss_weights = loss_weights.cuda()
    gt_quat = gt_quat.numpy()

    '''
    gt_inds = np.clip(np.floor((euler + np.pi) / (2 * np.pi) * nbins), 0, nbins - 1).astype(int)
    gt_quats = all_quats[gt_inds[:, 0], gt_inds[:, 1], gt_inds[:, 2]]
    gt = (
        torch.Tensor(
            np.clip(np.floor((euler + np.pi) / (2 * np.pi) * nbins), 0, nbins - 1)
        )
        .long()
        .cuda()
    )
    '''

    feats = model["cnn"](rgb_img_t).mean([2, 3])
    logits = model["predictor_mlp"](feats)

    exp_logits = torch.exp(logits)
    preds = exp_logits / torch.sum(exp_logits, dim=1, keepdims=True)

    loss = torch.sum(loss_weights * preds)
    loss.backward()

    opt.step()

    pred_quat_inds = torch.argmax(preds, dim=1).detach().cpu().numpy()
    pred_quats = gt_quats[pred_quat_inds]

    mean_pred_dist = np.mean(
        2 * np.arccos(
            np.clip(np.abs(np.sum(gt_quat * pred_quats, axis=-1)), 0, 1)
        )
    )
    gt_dists = 2 * np.arccos(
        np.clip(np.abs(np.sum(gt_quat[None] * gt_quat[:, None], axis=-1)), 0, 1)
    )
    mean_gt_dist = np.sum(gt_dists) / (len(gt_dists) ** 2 - len(gt_dists))
    break

    '''
    j = 3
    subplot(221)
    imshow(plt.imread('eye.png'))
    subplot(222)
    imshow(rgb_img[j])
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh('fuze.obj')
    verts = np.asarray(mesh.vertices)
    verts -= np.mean(verts, axis=0)
    verts *= 1.5

    verts_t = (gt_rotmat[j].numpy() @ verts.T).T
    # verts_cam = (inv(camera_pose) @ np.c_[verts, np.ones(len(verts))].T).T[:, :3]
    # verts_t_cam = (inv(camera_pose) @ np.c_[verts_t, np.ones(len(verts_t))].T).T[:, :3]

    a = [.2, 0, 0]
    b = scipy.spatial.transform.Rotation.from_quat(gt_quat[j]).as_matrix() @ a

    TODO
    visualize the closest gt_quat


    # gcf().add_subplot(223, projection='3d')
    # plot(verts_cam[:, 0], verts_cam[:, 1], verts_cam[:, 2], '.')
    # plot(verts_t_cam[:, 0], verts_t_cam[:, 1], verts_t_cam[:, 2], '.')
    # plot([0], [0], [0], '.')
    gcf().add_subplot(224, projection='3d')
    plot(*[[0, i] for i in a])
    plot(*[[0, i] for i in b])
    plot(verts[:, 0], verts[:, 1], verts[:, 2], '.')
    plot(verts_t[:, 0], verts_t[:, 1], verts_t[:, 2], '.')
    plot([.3], [0], [.35], '.')
    '''

    bin_edges = np.arange(len(gt_quats) + 1)
    if _wandb:
        wandb.log(
            {
                "mean gt dist": mean_gt_dist,
                "mean pred dist": mean_pred_dist,
                "loss": loss,
                "preds": wandb.Histogram(np_histogram=(torch.mean(preds, dim=0), bin_edges)),
            }
        )
