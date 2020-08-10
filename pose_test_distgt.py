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

sphere = o3d.geometry.TriangleMesh.create_sphere(resolution=20)
sphere_verts = np.asarray(sphere.vertices)
angles = np.arccos(np.dot(sphere_verts, [1, 0, 0]))
axes = np.cross(sphere_verts, [1, 0, 0])
rotvecs = axes / np.linalg.norm(axes, axis=-1, keepdims=True) * angles[:, None]
gt_rots = scipy.spatial.transform.Rotation.from_rotvec(rotvecs).inv()
# gt_rots = scipy.spatial.transform.Rotation.from_matrix(np.transpose(gt_rots.as_matrix(), (0, 2, 1)))
gt_quats = gt_rots.as_quat()
gt_eulers = gt_rots.as_euler("xyz")
gt_rotmats = gt_rots.as_matrix()

assert np.allclose(sphere_verts, gt_rotmats @ [1, 0, 0])


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

        ref_angles = np.arccos(
            np.clip(np.dot(sphere_verts, gt_rotmat @ [1, 0, 0]), 0, 1)
        )
        return rgb_img, rgb_img_t, gt_rotmat, gt_quat, ref_angles


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

cnn = torchvision.models.mobilenet_v2(pretrained=False).features

mlp = torch.nn.Sequential(
    FCLayer(1280, 1024),
    FCLayer(1024, 512),
    FCLayer(512, 256),
    FCLayer(256),
    FCLayer(256),
    torch.nn.Linear(256, len(gt_quats)),
)

model = torch.nn.ModuleDict({"cnn": cnn, "mlp": mlp}).cuda()
if _wandb:
    wandb.watch(model)

opt = torch.optim.Adam(
    [
        {"lr": 1e-3, "params": model["cnn"].parameters()},
        {"lr": 1e-3, "params": model["mlp"].parameters()},
    ]
)

crossentropy = torch.nn.CrossEntropyLoss()

for epoch in range(10):
    for batch in tqdm.tqdm(loader):
        opt.zero_grad()

        rgb_img, rgb_img_t, gt_rotmat, gt_quat, ref_angles = batch
        rgb_img_t = rgb_img_t.cuda()
        ref_angles = ref_angles.cuda()

        feats = model["cnn"](rgb_img_t).mean([2, 3])
        logits = model["mlp"](feats)

        gt_inds = np.argmin(
            np.arccos(
                np.clip(
                    np.sum(
                        (gt_rotmat.numpy()[:, None] @ [1, 0, 0])
                        * (gt_rotmats[None] @ [1, 0, 0]),
                        axis=-1,
                    ),
                    0,
                    1,
                )
            ),
            axis=-1,
        )
        gt_inds = torch.Tensor(gt_inds).long().cuda()
        target = torch.nn.functional.one_hot(gt_inds, num_classes=len(gt_quats)).float()

        loss = -torch.mean(
            torch.sum(
                torch.softmax(-ref_angles / 3, dim=-1)
                * torch.log_softmax(logits, dim=-1),
                dim=-1,
            )
        )
        # loss = -torch.mean(
        #     torch.sum(
        #         target * torch.log_softmax(logits, dim=-1), dim=-1
        #     )
        # )

        # crossentropy(logits, gt_inds)

        loss.backward()

        opt.step()

        exp_logits = torch.exp(logits)
        preds = exp_logits / torch.sum(exp_logits, dim=-1, keepdims=True)

        pred_rot_inds = torch.argmax(preds, dim=-1).detach().cpu().numpy()
        pred_rotmats = gt_rotmats[pred_rot_inds]

        pred_dists = np.arccos(
            np.clip(
                np.sum(
                    (pred_rotmats @ [1, 0, 0]) * (gt_rotmat.numpy() @ [1, 0, 0]),
                    axis=-1,
                ),
                0,
                1,
            )
        )
        mean_pred_dist = np.mean(pred_dists)

        gt_dists = np.arccos(
            np.clip(
                np.sum(
                    (gt_rotmat.numpy() @ [1, 0, 0])[None]
                    * (gt_rotmat.numpy() @ [1, 0, 0])[:, None],
                    axis=-1,
                ),
                0,
                1,
            )
        )
        mean_gt_dist = np.mean(gt_dists[(1 - np.tri(len(gt_dists))).astype(bool)])

        """
        j = 3
        rot = gt_rotmat[j, :3, :3].numpy()
        weights = loss_weights[j].cpu().numpy()
        angles = np.arccos(np.dot(sphere_verts, rot @ [1, 0, 0]))
        nearest_gt_rot = gt_rotmats[np.argmin(angles)]
        # colors = plt.cm.jet((angles - np.min(angles)) / (np.max(angles) - np.min(angles)))[:, :3]
        colors = plt.cm.jet((weights - np.min(weights)) / (np.max(weights) - np.min(weights)))[:, :3]
        a = [.3, 0, 0]
        b = rot @ a
        c = nearest_gt_rot @ a
        mesh = o3d.io.read_triangle_mesh('fuze.obj')
        verts = np.asarray(mesh.vertices)
        verts -= np.mean(verts, axis=0)
        verts *= 1.5
        verts_t = (rot @ verts.T).T
        subplot(221)
        imshow(plt.imread('eye.png'))
        subplot(222)
        imshow(rgb_img[j])
        gcf().add_subplot(223, projection='3d')
        plot(verts[:, 0], verts[:, 1], verts[:, 2], '.')
        plot(verts_t[:, 0], verts_t[:, 1], verts_t[:, 2], '.')
        plot([.3], [0], [.35], '.')
        plot(*[[0, i] for i in a])
        plot(*[[0, i] for i in b])
        plot(*[[0, i] for i in c])
        v = sphere_verts * .2
        gca().scatter(v[:, 0], v[:, 1], v[:, 2], c=colors, alpha=1)
    
        """

        bin_edges = np.arange(preds.shape[1] + 1)
        if _wandb:
            wandb.log(
                {
                    "mean gt dist": mean_gt_dist,
                    "mean pred dist": mean_pred_dist,
                    "loss": loss,
                    # "preds": wandb.Histogram(
                    #     np_histogram=(torch.mean(preds, dim=0), bin_edges)
                    # ),
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
