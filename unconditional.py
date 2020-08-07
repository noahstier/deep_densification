import functools
import glob
import importlib
import itertools
import os

import imageio
import numpy as np
import PIL.Image
import scipy.spatial
import torch
import torchvision
import tqdm
import unet
import wandb

import config

torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(1)


spec = importlib.util.spec_from_file_location(
    "colmap_reader", config.colmap_reader_script
)
colmap_reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(colmap_reader)


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


def interp_maps(maps, xy):
    x = xy[:, 0]
    y = xy[:, 1]

    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()

    x1 = torch.clamp_max(x0 + 1, maps.shape[3] - 1)
    y1 = torch.clamp_max(y0 + 1, maps.shape[2] - 1)

    batch_inds = torch.arange(maps.shape[0])

    assert torch.all(
        (y0 >= 0) & (y0 <= maps.shape[2] - 1) & (x0 >= 0) & (x0 <= maps.shape[3] - 1)
    )

    f_ll = maps[batch_inds, :, y0, x0]
    f_lr = maps[batch_inds, :, y0, x1]
    f_ul = maps[batch_inds, :, y1, x0]
    f_ur = maps[batch_inds, :, y1, x1]

    interped = (
        f_ll * ((x - x0) * (y - y0))[:, None]
        + f_lr * ((x1 - x) * (y - y0))[:, None]
        + f_ur * ((x1 - x) * (y1 - y))[:, None]
        + f_ul * ((x - x0) * (y1 - y))[:, None]
    )
    return interped


def positional_encoding(xyz, L):
    encoding = []
    for l in range(L):
        encoding.append(np.sin(2 ** l ** np.pi * xyz))
        encoding.append(np.cos(2 ** l ** np.pi * xyz))
    encoding = np.concatenate(encoding, axis=-1)
    return encoding


mlp = torch.nn.Sequential(
    FCLayer(35, 64), FCLayer(64), FCLayer(64), FCLayer(64), FCLayer(64), FCLayer(64),
)
occ_classifier = torch.nn.Sequential(
    FCLayer(64, 64), FCLayer(64, 32), FCLayer(32, 8), torch.nn.Linear(8, 1)
)
pointnet_mlp = torch.nn.Sequential(
    FCLayer(60, 64), FCLayer(64), FCLayer(64), FCLayer(64), FCLayer(64), FCLayer(64),
)

if config.wandb:
    wandb.init(project="deepmvs")

rgb_imgdir = os.path.join(config.house_dir, "imgs/color")
cat_imgdir = os.path.join(config.house_dir, "imgs/category")
fusion_npz = np.load(os.path.join(config.house_dir, "fusion.npz"))
sfm_imfile = os.path.join(config.house_dir, "sfm/sparse/auto/images.bin")
sfm_ptfile = os.path.join(config.house_dir, "sfm/sparse/auto/points3D.bin")

query_pts = fusion_npz["query_pts"]
query_tsdf = fusion_npz["query_tsdf"]
reprojection_samples = fusion_npz["reprojection_samples"]

ims = colmap_reader.read_images_binary(sfm_imfile)
pts = colmap_reader.read_points3d_binary(sfm_ptfile)

pts = {
    pt_id: pt for pt_id, pt in pts.items() if pt.error < 1 and len(pt.image_ids) >= 5
}

im_ids = sorted(ims.keys())
pt_ids = sorted(pts.keys())

sfm_xyz = np.stack([pts[i].xyz for i in pt_ids], axis=0)
sfm_rgb = np.stack([pts[i].rgb for i in pt_ids], axis=0)
kdtree = scipy.spatial.KDTree(sfm_xyz)

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

pil_imgs = [PIL.Image.open(os.path.join(rgb_imgdir, ims[i].name)) for i in im_ids]
imgs = np.stack([np.asarray(img) for img in pil_imgs], axis=0)
imgs_t = torch.stack([transform(img) for img in pil_imgs], dim=0).cuda()
imheight, imwidth, _ = imgs[0].shape

"""
cat_imgfiles = sorted(glob.glob(os.path.join(cat_imgdir, "*.pfm")))
cat_transform = torchvision.transforms.Resize(224, PIL.Image.NEAREST)
cat_imgs = np.stack(
    [cat_transform(PIL.Image.fromarray(imageio.imread(f)[::-1])) for f in cat_imgfiles],
    axis=0,
).astype(np.uint8)
cats = sorted(np.unique(cat_imgs).astype(np.uint8))
a = np.zeros(np.max(cats) + 1, dtype=np.uint8)
a[cats] = np.arange(len(cats))
cat_gt = a[cat_imgs]
cat_gt = torch.Tensor(cat_gt).long().cuda()
"""

intr_file = os.path.join(config.house_dir, "camera_intrinsics")
camera_intrinsic = np.loadtxt(intr_file)[0].reshape(3, 3)
camera_extrinsics = np.zeros((len(imgs), 4, 4))
for i, im_id in enumerate(im_ids):
    im = ims[im_id]
    qw, qx, qy, qz = im.qvec
    r = scipy.spatial.transform.Rotation.from_quat([qx, qy, qz, qw]).as_matrix().T
    t = np.linalg.inv(-r.T) @ im.tvec
    camera_extrinsics[i] = np.array(
        [[*r[0], t[0]], [*r[1], t[1]], [*r[2], t[2]], [0, 0, 0, 1]]
    )
query_neighborhood_radius = config.query_neighborhood_radius
query_pts_per_batch = config.query_pts_per_batch
imgs_per_anchor_pt = config.imgs_per_anchor_pt
anchor_pts_per_query = config.anchor_pts_per_query
n_pointnet_pts = config.n_pointnet_pts

"""
sfm_xyz = torch.Tensor(sfm_xyz).cuda()
keep = np.zeros(len(query_pts), dtype=bool)
for i in range(int(np.ceil(len(query_pts) / 10_000))):
    start = i * 10_000
    end = (i + 1) * 10_000
    q = torch.Tensor(query_pts[start:end]).cuda()
    dists = torch.norm(q[:, None] - sfm_xyz[None], dim=-1)
    keep[start:end] = (
        (torch.sum(dists < query_neighborhood_radius, dim=1) >= anchor_pts_per_query)
        .cpu()
        .numpy()
    )
sfm_xyz = sfm_xyz.cpu().numpy()
query_pts = query_pts[keep]
query_tsdf = query_tsdf[keep]
"""

model = torch.nn.Sequential(
    FCLayer(24, 256),
    FCLayer(256),
    FCLayer(256),
    FCLayer(256),
    FCLayer(256),
    FCLayer(256),
    FCLayer(256),
    FCLayer(256, 64),
    FCLayer(64, 16),
    torch.nn.Linear(16, 1),
).cuda()
model.train()

if config.wandb:
    wandb.watch(model)

optimizer = torch.optim.Adam(model.parameters())

# pos_weight = torch.Tensor([np.mean(query_tsdf > 0) / np.mean(query_tsdf < 0)])
# occ_loss_fn = torch.nn.BCELossWithLogits(pos_weight=pos_weight).cuda()
occ_loss_fn = torch.nn.BCELoss().cuda()
cat_loss_fn = torch.nn.CrossEntropyLoss().cuda()

query_pts -= np.array([3.75210289e01, 4.46640313e-02, 4.68986430e01])
query_pts /= np.array([7.52712135, 2.74440703, 6.09322482])
query_pts = query_pts * 2 - 1

query_pts = positional_encoding(query_pts, L=4)

step = -1
for epoch in range(50):
    print("epoch {}".format(epoch))
    indices = np.arange(len(query_pts), dtype=int)
    np.random.shuffle(indices)
    indices = iter(indices)

    for _ in tqdm.trange(len(query_pts) // query_pts_per_batch):
        optimizer.zero_grad()

        query_inds = np.array([next(indices) for _ in range(query_pts_per_batch)])
        query_xyz = query_pts[query_inds]
        query_occ = query_tsdf[query_inds] < 0.001

        m = np.mean(query_occ)
        if m < 0.1 or m > 0.9:
            continue

        mlp_input = torch.Tensor(query_xyz).cuda()
        logits = model(mlp_input)[..., 0]

        preds = torch.sigmoid(logits)

        query_occ = torch.Tensor(query_occ).cuda()

        pos_inds = query_occ.bool()
        neg_inds = ~pos_inds
        pos_acc = torch.sum((preds > 0.5) & pos_inds) / pos_inds.sum().float()
        neg_acc = torch.sum((preds < 0.5) & neg_inds) / neg_inds.sum().float()

        true_pos = torch.sum((preds > 0.5) & pos_inds).float()
        all_predicted_pos = torch.sum(preds > 0.5)
        all_actual_pos = torch.sum(pos_inds)
        precision = true_pos / all_predicted_pos
        recall = true_pos / all_actual_pos

        # occ_loss = occ_loss_fn(preds, query_occ)
        # cat_loss = cat_loss_fn(cat_logits, cat_gt[visible_img_inds])

        pos_loss = occ_loss_fn(preds[pos_inds], query_occ[pos_inds])
        neg_loss = occ_loss_fn(preds[neg_inds], query_occ[neg_inds])

        # loss = pos_loss + neg_loss
        loss = occ_loss_fn(preds, query_occ)

        loss.backward()
        optimizer.step()

        step += 1
        if config.wandb:
            wandb.log(
                {
                    "pos loss": pos_loss.item(),
                    "neg loss": neg_loss.item(),
                    # "loss": loss.item(),
                    # "cat_loss": cat_loss.item(),
                    "logits": wandb.Histogram(logits.detach().cpu().numpy()),
                    "preds": wandb.Histogram(preds.detach().cpu().numpy()),
                    "occ": wandb.Histogram(query_occ.cpu().numpy()),
                    "pos_acc": pos_acc.item(),
                    "neg_acc": neg_acc.item(),
                    "precision": precision.item(),
                    "recall": recall.item(),
                },
                step=step,
            )

    name = "pos4"
    torch.save(model, os.path.join("models", name))
