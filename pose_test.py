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

nbins = 36

x, y, z = np.linspace(np.zeros(3), np.ones(3) * 2 * np.pi, nbins, axis=1)
xx, yy, zz = np.meshgrid(x, y, z)
all_quats = scipy.spatial.transform.Rotation.from_euler('xyz', np.c_[xx.flatten(), yy.flatten(), zz.flatten()]).as_quat()


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

        gt_dists = 2 * np.arccos(np.abs(np.sum(gt_quat * all_quats, axis=-1)))
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

s = np.sqrt(2) / 2
camera_pose = np.array(
    [
        [0.0, -s, s, 0.3],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, s, s, 0.35],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


cnn = torchvision.models.mobilenet_v2(pretrained=True).features

predictor_mlp = torch.nn.Sequential(
    FCLayer(1280, 1024),
    FCLayer(1024, 512),
    FCLayer(512),
    FCLayer(512),
    torch.nn.Linear(512, nbins * 3),
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

    gt_euler = scipy.spatial.transform.Rotation.from_quat(gt_quat).as_euler('xyz')

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

    angle_logits = logits.reshape(batch_size, 3, nbins)
    exp_angle_logits = torch.exp(angle_logits)
    angle_preds = exp_angle_logits / torch.sum(exp_angle_logits, dim=-1, keepdims=True)

    j = 0
    view_preds = []
    for j in range(batch_size):
        xx, yy, zz = torch.meshgrid(angle_preds[j, 0], angle_preds[j, 1], angle_preds[j, 2])
        view_preds.append(torch.prod(torch.stack((xx.flatten(), yy.flatten(), zz.flatten()), axis=0), dim=0))
    view_preds = torch.stack(view_preds, axis=0)

    '''
    _logits = logits.detach().cpu().numpy().reshape(batch_size, 3, -1)
    preds = np.exp(_logits) / np.sum(np.exp(_logits), axis=-1, keepdims=True)
    preds = np.argmax(preds, axis=-1) * 2 * np.pi / nbins
    pred_quats = scipy.spatial.transform.Rotation.from_euler("xyz", preds).as_quat()
    pred_quat_dists = 2 * np.arccos(np.abs(np.sum(pred_quats * gt_quats, axis=1)))
    '''


    loss = torch.sum(loss_weights * view_preds)
    loss.backward()

    opt.step()

    pred_quat_inds = torch.argmax(view_preds, dim=1).detach().cpu().numpy()
    pred_quats = all_quats[pred_quat_inds]

    mean_pred_dist = np.mean(
        2 * np.arccos(
            np.clip(np.abs(np.sum(gt_quat * pred_quats, axis=-1)), 0, 1)
        )
    )
    gt_dists = 2 * np.arccos(
        np.clip(np.abs(np.sum(gt_quat[None] * gt_quat[:, None], axis=-1)), 0, 1)
    )
    mean_gt_dist = np.sum(gt_dists) / (len(gt_dists) ** 2 - len(gt_dists))

    '''
    j = 4
    subplot(221)
    imshow(plt.imread('eye.png'))
    subplot(222)
    imshow(rgb_img[j])
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh('fuze.obj')
    verts = np.asarray(mesh.vertices)
    verts_t = (gt_rotmat[j].numpy() @ verts.T).T
    verts_cam = (inv(camera_pose) @ np.c_[verts, np.ones(len(verts))].T).T[:, :3]
    verts_t_cam = (inv(camera_pose) @ np.c_[verts_t, np.ones(len(verts_t))].T).T[:, :3]
    gcf().add_subplot(223, projection='3d')
    plot(verts_cam[:, 0], verts_cam[:, 1], verts_cam[:, 2], '.')
    plot(verts_t_cam[:, 0], verts_t_cam[:, 1], verts_t_cam[:, 2], '.')
    plot([0], [0], [0], '.')
    gcf().add_subplot(224, projection='3d')
    plot(verts[:, 0], verts[:, 1], verts[:, 2], '.')
    plot(verts_t[:, 0], verts_t[:, 1], verts_t[:, 2], '.')
    plot([.3], [0], [.35], '.')
    '''

    break
    v_t = gt @ v
    v_t_c = cam @ gt @ v

    gt_x = np.zeros(nbins)
    gt_y = np.zeros(nbins)
    gt_z = np.zeros(nbins)
    gt_inds = np.floor((gt_euler + np.pi) / (np.pi * 2) * nbins).astype(int)
    gt_inds = np.clip(gt_inds, 0, nbins - 1)
    gt_x[gt_inds[:, 0]] = 1
    gt_y[gt_inds[:, 0]] = 1
    gt_z[gt_inds[:, 0]] = 1


    bin_edges = np.linspace(0, 360, nbins + 1)
    angle_preds = np.mean(angle_preds.detach().cpu().numpy(), axis=0)
    if _wandb:
        wandb.log(
            {
                "mean gt dist": mean_gt_dist,
                "mean pred dist": mean_pred_dist,
                "loss": loss,
                "pred x": wandb.Histogram(np_histogram=(angle_preds[0], bin_edges)),
                "pred y": wandb.Histogram(np_histogram=(angle_preds[1], bin_edges)),
                "pred z": wandb.Histogram(np_histogram=(angle_preds[2], bin_edges)),
                "gt x": wandb.Histogram(np_histogram=(gt_x, bin_edges)),
                "gt y": wandb.Histogram(np_histogram=(gt_y, bin_edges)),
                "gt z": wandb.Histogram(np_histogram=(gt_z, bin_edges)),
            }
        )
