_wandb = False

import glob
import logging
import os

import h5py
import PIL.Image
import numpy as np
import open3d as o3d
import scipy.spatial
import torch
import torchvision
import trimesh
import tqdm

trimesh.constants.log.setLevel(logging.ERROR)

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
gt_rots = scipy.spatial.transform.Rotation.from_rotvec(rotvecs).inv()
# gt_rots = scipy.spatial.transform.Rotation.from_matrix(np.transpose(gt_rots.as_matrix(), (0, 2, 1)))
gt_quats = gt_rots.as_quat()
gt_eulers = gt_rots.as_euler("xyz")
gt_rotmats = gt_rots.as_matrix()
gt_rotmats_cuda = torch.Tensor(gt_rotmats).cuda()

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

        self.mesh = trimesh.load("fuze.obj")
        self.verts = np.asarray(self.mesh.vertices)
        self.verts -= np.mean(self.verts, axis=0)
        self.verts *= 1.5
        self.maxnorm = np.linalg.norm(np.max(self.verts, axis=0))
        self.n_normal = 200
        self.n_uniform = 56

    def __len__(self):
        return len(self.rgb_dset)

    def __getitem__(self, index):
        rgb_img = self.rgb_dset[index]
        pil_img = PIL.Image.fromarray(rgb_img)
        rgb_img_t = transform(pil_img)
        pose = self.pose_dset[index]

        x = np.random.normal(0, 1, size=(self.n_uniform, 5))
        uniform_query_pts = (
            x[:, :3] / np.linalg.norm(x, axis=-1, keepdims=True) * self.maxnorm * 1.1
        )
        normal_query_pts = self.mesh.sample(self.n_normal) + np.random.normal(
            0, 0.05, size=(self.n_normal, 3)
        )
        query_pts = np.concatenate((uniform_query_pts, normal_query_pts), axis=0)
        query_occ = self.mesh.contains(query_pts)
        query_pts = (pose[:3, :3] @ query_pts.T).T
        rot = scipy.spatial.transform.Rotation.from_matrix(pose[:3, :3])
        gt_rotmat = rot.as_matrix()
        gt_quat = rot.as_quat()

        if np.any(np.isnan(gt_quat)):
            return self[np.random.randint(0, len(self))]

        ref_angles = np.arccos(
            np.clip(np.dot(sphere_verts, gt_rotmat @ [1, 0, 0]), 0, 1)
        )
        return (
            rgb_img,
            rgb_img_t,
            gt_rotmat,
            gt_quat,
            ref_angles,
            query_pts.astype(np.float32),
            query_occ,
        )


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

pose_mlp = torch.nn.Sequential(
    FCLayer(1280, 1024),
    FCLayer(1024, 512),
    FCLayer(512, 256),
    FCLayer(256),
    FCLayer(256),
    torch.nn.Linear(256, len(gt_quats)),
)
occ_mlp = torch.nn.Sequential(
    # FCLayer(1280 + 1024, 1024),
    FCLayer(1024, 512),
    FCLayer(512, 256),
    FCLayer(256, 128),
    FCLayer(128, 64),
    FCLayer(64, 16),
    torch.nn.Linear(16, 1),
)

query_encoder = torch.nn.Sequential(
    FCLayer(3, 16),
    FCLayer(16, 64),
    FCLayer(64, 128),
    FCLayer(128, 256),
    FCLayer(256, 512),
    FCLayer(512, 1024),
)
# occ_mlp = torch.nn.Sequential(
#     FCLayer(3, 16),
#     FCLayer(16, 64),
#     FCLayer(64),
#     FCLayer(64),
#     FCLayer(64),
#     FCLayer(64, 16),
#     torch.nn.Linear(16, 1)
# )

model = torch.nn.ModuleDict(
    {
        "cnn": cnn,
        "pose_mlp": pose_mlp,
        "occ_mlp": occ_mlp,
        "query_encoder": query_encoder,
    }
).cuda()
if _wandb:
    wandb.watch(model)

opt = torch.optim.Adam(
    [
        {"lr": 1e-3, "params": model["cnn"].parameters()},
        {"lr": 1e-3, "params": model["pose_mlp"].parameters()},
        {"lr": 1e-3, "params": model["occ_mlp"].parameters()},
        {"lr": 1e-3, "params": model["query_encoder"].parameters()},
    ]
)

bce = torch.nn.BCEWithLogitsLoss()

step = 0
for epoch in range(10):
    for batch in tqdm.tqdm(loader):
        opt.zero_grad()

        rgb_img, rgb_img_t, gt_rotmat, gt_quat, ref_angles, query_pts, query_occ = batch
        rgb_img_t = rgb_img_t.cuda()
        ref_angles = ref_angles.cuda()
        query_pts = query_pts.cuda()
        query_occ = query_occ.cuda().float()
        gt_rotmat_cuda = gt_rotmat.float().cuda()

        feats = model["cnn"](rgb_img_t).mean([2, 3])
        logits = model["pose_mlp"](feats)

        preds = torch.softmax(logits, dim=-1)
        cum_preds = torch.cumsum(preds, dim=-1)
        inds = torch.sum(torch.rand(len(preds)).cuda()[:, None] > cum_preds, dim=-1)
        pred_rotmats = gt_rotmats_cuda[inds]
        wsum_rotmats = (preds[..., None, None] * gt_rotmats_cuda).mean(1)

        # query_pts_t = (pred_rotmats @ query_pts.transpose(1, 2).cuda().float()).transpose(1,2)
        query_pts_t = (
            gt_rotmat_cuda.inverse() @ query_pts.transpose(1, 2).cuda().float()
        ).transpose(1, 2)
        # mlp_input = torch.cat(
        #     (model['query_encoder'](query_pts_t), feats[:, None].repeat((1, query_pts.shape[1], 1))), dim=-1
        # )
        mlp_input = model["query_encoder"](query_pts_t)
        occ_logits = model["occ_mlp"](mlp_input)[..., 0]

        pos_loss = bce(occ_logits[query_occ.bool()], query_occ[query_occ.bool()])
        neg_loss = bce(occ_logits[~(query_occ.bool())], query_occ[~(query_occ.bool())])
        loss = pos_loss + neg_loss
        loss.backward()

        """
        j = 2
        plot3()
        q = query_pts_t.detach().cpu().numpy()[j]
        o = query_occ.cpu().numpy().astype(int)[j]
        v = dset.verts
        v_t = (gt_rotmat[j].inverse().numpy() @ v.T).T
        colors = np.array([[1, 0, 0], [0, 0, 1]])[o]
        gca().scatter(q[:, 0], q[:, 1], q[:, 2], c=colors)
        plot(v[:, 0], v[:, 1], v[:, 2], 'k.')
        """

        opt.step()

        # pred_rot_inds = torch.argmax(preds, dim=-1).detach().cpu().numpy()
        # pred_rotmats = gt_rotmats[pred_rot_inds]

        # pred_dists = np.arccos(
        #     np.clip(
        #         np.sum(
        #             (pred_rotmats @ [1, 0, 0]) * (gt_rotmat.numpy() @ [1, 0, 0]), axis=-1
        #         ),
        #         0,
        #         1,
        #     )
        # )
        # mean_pred_dist = np.mean(pred_dists)

        # gt_dists = np.arccos(
        #     np.clip(
        #         np.sum(
        #             (gt_rotmat.numpy() @ [1, 0, 0])[None]
        #             * (gt_rotmat.numpy() @ [1, 0, 0])[:, None],
        #             axis=-1,
        #         ),
        #         0,
        #         1,
        #     )
        # )
        # mean_gt_dist = np.mean(gt_dists[(1 - np.tri(len(gt_dists))).astype(bool)])

        """
        j = 3

        rot = gt_rotmat[j, :3, :3].numpy()
        angles = np.arccos(np.dot(sphere_verts, rot @ [1, 0, 0]))
        nearest_gt_rot = gt_rotmats[np.argmin(angles)]

        loss_weights = torch.softmax(-ref_angles / 3, dim=-1)[j].cpu().numpy()
        loss_weights -= np.min(loss_weights)
        loss_weights /= np.max(loss_weights)
        colors = plt.cm.jet(loss_weights)[:, :3]
        a = [.3, 0, 0]
        b = rot @ a
        c = nearest_gt_rot @ a
        verts = dset.verts
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
        gcf().add_subplot(224, projection='3d')
        plot(verts_t[:, 0], verts_t[:, 1], verts_t[:, 2], '.', markersize=1, color='k')
        q = query_pts[j].cpu().numpy()
        query_colors = np.array([[1, 0, 0], [0, 0, 1]])[query_occ[j].int().cpu().numpy()]
        gca().scatter(q[:, 0], q[:, 1], q[:, 2], alpha=1, c=query_colors, s=1)
        """
        if step % 101 == 0:
            ...

        bin_edges = np.arange(preds.shape[1] + 1)
        if _wandb:
            wandb.log(
                {
                    # "mean gt dist": mean_gt_dist,
                    # "mean pred dist": mean_pred_dist,
                    "loss": loss,
                    "pos_loss": pos_loss,
                    "neg_loss": neg_loss,
                    "preds": wandb.Histogram(
                        np_histogram=(torch.mean(preds, dim=0), bin_edges)
                    ),
                },
                step=step,
            )
        step += 1


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
