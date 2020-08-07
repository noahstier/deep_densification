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

minbounds = np.array([44, 0, 47])
maxbounds = np.array([45, 2, 50])
inds = (
    (query_pts[:, 0] > minbounds[0])
    & (query_pts[:, 0] < maxbounds[0])
    & (query_pts[:, 1] > minbounds[1])
    & (query_pts[:, 1] < maxbounds[1])
    & (query_pts[:, 2] > minbounds[2])
    & (query_pts[:, 2] < maxbounds[2])
)
query_pts = query_pts[inds]
query_tsdf = query_tsdf[inds]

model = torch.nn.ModuleDict(
    {
        "cnn": torchvision.models.mobilenet_v2(pretrained=True).features[:7],
        # "cnn": unet.UNet(3, 32),
        "mlp": mlp,
        "pointnet_mlp": pointnet_mlp,
        "occ_classifier": occ_classifier,
        # "query_embedder": torch.nn.Sequential(
        #     FCLayer(3, 16), FCLayer(16, 32), FCLayer(32, 64)
        # ),
        # "img_classifier": torch.nn.Conv2d(32, len(cats), 1),
    }
).cuda()

model.train()

if config.wandb:
    wandb.watch(model)

feat_height, feat_width = model["cnn"](imgs_t[:1]).shape[2:]

optimizer = torch.optim.Adam(
    [
        {"lr": 1e-4, "params": model["cnn"].parameters()},
        {"lr": 1e-3, "params": model["mlp"].parameters()},
        {"lr": 1e-3, "params": model["pointnet_mlp"].parameters()},
        {"lr": 1e-3, "params": model["occ_classifier"].parameters()},
    ]
)

# pos_weight = torch.Tensor([np.mean(query_tsdf > 0) / np.mean(query_tsdf < 0)])
# occ_loss_fn = torch.nn.BCELossWithLogits(pos_weight=pos_weight).cuda()
occ_loss_fn = torch.nn.BCELoss().cuda()
cat_loss_fn = torch.nn.CrossEntropyLoss().cuda()

step = -1
for epoch in range(10):
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
        if m < 0.2 or m > 0.8:
            continue

        anchor_dists, anchor_pt_inds = kdtree.query(
            query_xyz, k=32, distance_upper_bound=query_neighborhood_radius
        )
        anchor_pt_inds = np.stack(
            [
                np.random.choice(
                    inds[dists < query_neighborhood_radius],
                    size=anchor_pts_per_query,
                    replace=False,
                )
                for inds, dists in zip(anchor_pt_inds, anchor_dists)
            ],
            axis=0,
        )

        anchor_pt_ids = np.array(pt_ids)[anchor_pt_inds]
        anchor_xyz = sfm_xyz[anchor_pt_inds]

        _, local_sfm_inds = kdtree.query(anchor_xyz, k=n_pointnet_pts)
        local_sfm_xyz = sfm_xyz[local_sfm_inds]

        anchor_img_inds = np.zeros(
            (*anchor_pt_ids.shape, imgs_per_anchor_pt), dtype=int
        )
        anchor_uv = np.zeros((*anchor_pt_ids.shape, imgs_per_anchor_pt, 2))
        for i in range(anchor_pt_ids.shape[0]):
            for j in range(anchor_pt_ids.shape[1]):
                anchor_pt = pts[anchor_pt_ids[i, j]]
                inds = np.random.choice(
                    np.arange(len(anchor_pt.image_ids)),
                    size=imgs_per_anchor_pt,
                    replace=False,
                )
                anchor_img_ids = anchor_pt.image_ids[inds]
                point2D_idxs = anchor_pt.point2D_idxs[inds]
                anchor_uv[i, j] = [
                    ims[img_id].xys[point2D_idx]
                    for img_id, point2D_idx in zip(anchor_img_ids, point2D_idxs)
                ]
                for k in range(imgs_per_anchor_pt):
                    anchor_img_inds[i, j, k] = im_ids.index(anchor_img_ids[k])

        poses = camera_extrinsics[anchor_img_inds]

        anchor_uv_homog = np.concatenate(
            (anchor_uv, np.ones((*anchor_uv.shape[:-1], 1))), axis=-1
        )
        pixel_vecs = anchor_uv_homog @ np.linalg.inv(camera_intrinsic).T
        pixel_vecs /= np.linalg.norm(pixel_vecs, axis=-1, keepdims=True)
        quat_xyz = np.stack(
            (-pixel_vecs[..., 1], pixel_vecs[..., 0], np.zeros(pixel_vecs.shape[:-1])),
            axis=-1,
        )
        quat_w = pixel_vecs[..., 2] + 1
        quats = np.concatenate((quat_xyz, quat_w[..., None]), axis=-1)
        r2 = (
            scipy.spatial.transform.Rotation.from_quat(np.reshape(quats, (-1, 4)))
            .as_matrix()
            .reshape(*quats.shape[:3], 3, 3)
        )
        t = poses[..., :3, 3]
        r1 = poses[..., :3, :3]
        corrected_poses = np.tile(np.eye(4), (*poses.shape[:3], 1, 1))
        corrected_poses[..., :3, :3] = r1 @ r2
        corrected_poses[..., :3, 3] = t

        fakeposes = np.tile(np.eye(4), (*poses.shape[:3], 1, 1))

        # poses = poses
        # poses = corrected_poses
        # poses = fakeposes

        imshape = np.array([imgs.shape[2], imgs.shape[1]])
        featshape = np.array([feat_width, feat_height])
        anchor_uv_t = (
            anchor_uv / imshape[None, None, None] * featshape[None, None, None]
        )

        query_xyz_cam = np.stack(
            [
                (np.linalg.inv(poses[i]) @ [*query_xyz[i], 1])[..., :3]
                for i in range(query_pts_per_batch)
            ],
            axis=0,
        )

        anchor_xyz_cam = np.zeros(
            (query_pts_per_batch, anchor_pts_per_query, imgs_per_anchor_pt, 3)
        )
        for i in range(query_pts_per_batch):
            for j in range(anchor_pts_per_query):
                anchor_xyz_cam[i, j] = (
                    np.linalg.inv(poses[i, j]) @ [*anchor_xyz[i, j], 1]
                )[:, :3]

        rel_query_xyz = (query_xyz_cam - anchor_xyz_cam) / query_neighborhood_radius
        rel_query_pts = positional_encoding(rel_query_xyz, L=10)
        rel_query_pts = torch.Tensor(rel_query_pts).cuda()

        sfm_xyz_cam = (np.linalg.inv(poses) @ np.c_[sfm_xyz, np.ones(len(sfm_xyz))].T)[
            ..., :3, :
        ].transpose(0, 1, 2, 4, 3)
        nn_sfm_xyz_cam = np.zeros(
            (
                query_pts_per_batch,
                anchor_pts_per_query,
                imgs_per_anchor_pt,
                n_pointnet_pts,
                3,
            )
        )
        nn_sfm_rgb = np.zeros(
            (query_pts_per_batch, anchor_pts_per_query, n_pointnet_pts, 3,)
        )
        for i in range(query_pts_per_batch):
            for j in range(anchor_pts_per_query):
                nn_sfm_xyz_cam[i, j] = sfm_xyz_cam[i, j][:, local_sfm_inds[i, j]]
                nn_sfm_rgb[i, j] = sfm_rgb[local_sfm_inds[i, j]]

        rel_nn_sfm_xyz = (
            nn_sfm_xyz_cam - anchor_xyz_cam[:, :, :, None, :]
        ) / query_neighborhood_radius
        rel_nn_sfm_pts = positional_encoding(rel_nn_sfm_xyz, L=10)
        rel_nn_sfm_pts = torch.Tensor(rel_nn_sfm_pts).cuda()
        anchor_uv_t = torch.Tensor(anchor_uv_t).cuda()

        pointnet_feats, _ = torch.max(model["pointnet_mlp"](rel_nn_sfm_pts), dim=3)

        shape = anchor_img_inds.shape
        img_feats = model["cnn"](imgs_t[anchor_img_inds.flatten()])
        # cat_logits = model["img_classifier"](img_feats[:, :32])

        pixel_feats = interp_maps(img_feats, anchor_uv_t.reshape(-1, 2))
        pixel_feats = pixel_feats.reshape(*shape, -1)

        rel_query_pts = (
            query_xyz[:, None, None] - anchor_xyz[:, None]
        ) / query_neighborhood_radius
        # rel_query_pts = positional_encoding(rel_query_pts, L=10)
        rel_query_pts = torch.Tensor(rel_query_pts).cuda()

        # mlp_input = torch.cat((rel_query_pts, pixel_feats, pointnet_feats), dim=3)
        mlp_input = torch.cat((rel_query_pts, pixel_feats), dim=3)
        query_pt_feats = model["mlp"](mlp_input)

        query_pt_feats, _ = torch.max(query_pt_feats, dim=1)
        query_pt_feats, _ = torch.max(query_pt_feats, dim=1)

        logits = model["occ_classifier"](query_pt_feats)[..., 0]

        """
        gcf().add_subplot(121, projection='3d')
        plot(sfm_xyz_cam[0, 0, 0, :, 0], sfm_xyz_cam[0, 0, 0, :, 1], sfm_xyz_cam[0, 0, 0, :, 2], 'k.', markersize=1)
        plot([0, anchor_xyz_cam[0, 0, 0, 0]], [0, anchor_xyz_cam[0, 0, 0, 1]], [0, anchor_xyz_cam[0, 0, 0, 2]])
        plot([query_xyz_cam[0, 0, 0, 0]], [query_xyz_cam[0, 0, 0, 1]], [query_xyz_cam[0, 0, 0, 2]], 'b.', markersize=10)

        subplot(122)
        imshow(imgs[anchor_img_inds[0, 0, 0]])
        plot(anchor_uv[0, 0, 0, 0], anchor_uv[0, 0, 0, 1], 'b.')
        axis('off')

        tight_layout()
        
        """

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

        loss = pos_loss + neg_loss

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

    name = wandb.run.name if wandb.run else "noname"
    torch.save(model, os.path.join("models", name))
