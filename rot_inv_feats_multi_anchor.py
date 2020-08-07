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

colmap_reader_script = "/home/noah/Documents/colmap/scripts/python/read_write_model.py"
spec = importlib.util.spec_from_file_location("colmap_reader", colmap_reader_script)
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
        x = self.fc(inputs)
        if self.use_bn:
            if x.ndim > 2:
                x = self.bn(x.transpose(1, -1)).transpose(1, -1)
            else:
                x = self.bn(x)
        x = torch.relu(x)
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


mlp = torch.nn.Sequential(
    FCLayer(144, 64), FCLayer(64), FCLayer(64), FCLayer(64), FCLayer(64), FCLayer(64),
)
occ_classifier = torch.nn.Sequential(
    FCLayer(64, 64), FCLayer(64, 32), FCLayer(32, 8), torch.nn.Linear(8, 1)
)
pointnet_mlp = torch.nn.Sequential(
    FCLayer(3, 64), FCLayer(64), FCLayer(64), FCLayer(64), FCLayer(64), FCLayer(64),
)


_wandb = False
if _wandb:
    wandb.init(project="deepmvs")


house_dir = "/home/noah/Documents/scenecompletion/onetpp/data_preprocessing/suncg/output/000514ade3bcc292a613a4c2755a5050"

rgb_imgdir = os.path.join(house_dir, "imgs/color")
cat_imgdir = os.path.join(house_dir, "imgs/category")
fusion_npz = np.load(os.path.join(house_dir, "fusion.npz"))
sfm_imfile = os.path.join(house_dir, "sfm/sparse/auto/images.bin")
sfm_ptfile = os.path.join(house_dir, "sfm/sparse/auto/points3D.bin")

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

intr_file = os.path.join(house_dir, "camera_intrinsics")
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

query_neighborhood_radius = 0.2
query_pts_per_batch = 128
imgs_per_anchor_pt = 2
anchor_pts_per_batch = 4

model = torch.nn.ModuleDict(
    {
        # 'cnn': torchvision.models.mobilenet_v2(pretrained=True).features,
        "cnn": unet.UNet(3, 32),
        "mlp": mlp,
        "pointnet_mlp": pointnet_mlp,
        "occ_classifier": occ_classifier,
        "query_embedder": torch.nn.Sequential(
            FCLayer(3, 16), FCLayer(16, 32), FCLayer(32, 64)
        ),
        "img_classifier": torch.nn.Conv2d(32, len(cats), 1),
    }
).cuda()

model.train()

if _wandb:
    wandb.watch(model)

feat_height, feat_width = model["cnn"](imgs_t[:1]).shape[2:]

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# {"lr": 1e-3, "params": model['cnn'].parameters()},
# {"lr": 1e-3, "params": model['mlp'].parameters()},
# {"lr": 1e-3, "params": model['occ_classifier'].parameters()},
# {"lr": 1e-3, "params": model['query_embedder'].parameters()}
# ])

# pos_weight = torch.Tensor([np.mean(query_tsdf > 0) / np.mean(query_tsdf < 0)])
# occ_loss_fn = torch.nn.BCELossWithLogits(pos_weight=pos_weight).cuda()
occ_loss_fn = torch.nn.BCELoss().cuda()
cat_loss_fn = torch.nn.CrossEntropyLoss().cuda()

indices = np.arange(len(pt_ids), dtype=int)
np.random.seed(0)
np.random.shuffle(indices)
indices = iter(indices)

for step in tqdm.trange(3000):
    optimizer.zero_grad()

    for n in itertools.count():
        index = next(indices)
        dists, pt_inds = kdtree.query(
            pts[pt_ids[index]].xyz, k=32, distance_upper_bound=0.2
        )
        dists = dists[1:]
        pt_inds = pt_inds[1:]
        pt_inds = pt_inds[~np.isinf(dists)]
        pt_inds = [
            i for i in pt_inds if len(pts[pt_ids[i]].image_ids) >= imgs_per_anchor_pt
        ]
        if len(pt_inds) < anchor_pts_per_batch - 1:
            continue
        pt_inds = np.random.choice(
            pt_inds, size=anchor_pts_per_batch - 1, replace=False
        )
        anchor_pts = [pts[pt_ids[i]] for i in [index, *pt_inds]]

        query_pt_inds = np.argwhere(
            np.linalg.norm(query_pts - anchor_pts[0].xyz, axis=1)
            < query_neighborhood_radius
        ).flatten()

        if len(query_pt_inds) < query_pts_per_batch:
            continue

        query_pt_inds = np.random.choice(
            query_pt_inds, size=query_pts_per_batch, replace=False
        )
        query_xyz = query_pts[query_pt_inds]
        query_occ = query_tsdf[query_pt_inds] < 0.001

        m = np.mean(query_occ)
        if m < 0.2 or m > 0.8:
            continue

        break

    _, local_sfm_inds = kdtree.query(query_xyz, k=256)
    query_occ = torch.Tensor(query_occ).cuda()

    visible_img_ids = [pt.image_ids for pt in anchor_pts]
    visible_img_inds = [
        np.array([im_ids.index(i) for i in ids]) for ids in visible_img_ids
    ]
    uv = [
        np.stack(
            [
                ims[im_id].xys[point2D_idx]
                for im_id, point2D_idx in zip(
                    anchor_pt.image_ids, anchor_pt.point2D_idxs
                )
            ],
            axis=0,
        )
        for anchor_pt in anchor_pts
    ]
    uv_t = [
        _uv / [imgs.shape[2], imgs.shape[1]] * [feat_width, feat_height] for _uv in uv
    ]
    uv_t = [torch.Tensor(_uv_t).cuda() for _uv_t in uv_t]

    for i in range(len(visible_img_ids)):
        if len(visible_img_ids[i]) > imgs_per_anchor_pt:
            inds = np.arange(len(visible_img_ids[i]), dtype=int)
            inds = np.random.choice(inds, size=imgs_per_anchor_pt, replace=False)
            visible_img_ids[i] = visible_img_ids[i][inds]
            visible_img_inds[i] = visible_img_inds[i][inds]
            uv[i] = uv[i][inds]
            uv_t[i] = uv_t[i][inds]

    visible_img_ids = np.concatenate(visible_img_ids, axis=0)
    visible_img_inds = np.concatenate(visible_img_inds, axis=0)
    uv = np.concatenate(uv, axis=0)
    uv_t = torch.cat(uv_t, dim=0)

    anchor_xyz = np.concatenate(
        [np.stack((pt.xyz,) * imgs_per_anchor_pt, axis=0) for pt in anchor_pts], axis=0
    )

    poses = camera_extrinsics[visible_img_inds]

    query_xyz_cam = (
        np.linalg.inv(poses) @ np.c_[query_xyz, np.ones(len(query_xyz))].T
    ).transpose(2, 0, 1)
    anchor_xyz_cam = np.stack(
        [np.linalg.inv(poses[i]) @ [*anchor_xyz[i], 1] for i in range(len(poses))],
        axis=0,
    )
    rel_query_xyz = (query_xyz_cam - anchor_xyz_cam[None])[
        ..., :3
    ] / query_neighborhood_radius
    rel_query_xyz = torch.Tensor(rel_query_xyz).cuda()
    query_embedding = model["query_embedder"](rel_query_xyz)

    sfm_xyz_cam = (
        np.linalg.inv(poses) @ np.c_[sfm_xyz, np.ones(len(sfm_xyz))].T
    ).transpose(2, 0, 1)
    nn_sfm_xyz_cam = sfm_xyz_cam[local_sfm_inds, :, :]
    rel_nn_sfm_xyz = (nn_sfm_xyz_cam - anchor_xyz_cam[None, None])[
        ..., :3
    ] / query_neighborhood_radius
    rel_nn_sfm_xyz = torch.Tensor(rel_nn_sfm_xyz).cuda()
    shape = rel_nn_sfm_xyz.shape[:3]
    rel_nn_sfm_xyz = rel_nn_sfm_xyz.reshape(
        rel_nn_sfm_xyz.shape[0],
        rel_nn_sfm_xyz.shape[1] * rel_nn_sfm_xyz.shape[2],
        rel_nn_sfm_xyz.shape[3],
    )
    pointnet_feats, _ = torch.max(
        model["pointnet_mlp"](rel_nn_sfm_xyz).reshape(*shape, 64), dim=1
    )

    img_feats = model["cnn"](imgs_t[visible_img_inds])
    # cat_logits = model["img_classifier"](img_feats[:, :32])

    pixel_feats = interp_maps(img_feats, uv_t)
    pixel_feats = pixel_feats[None].repeat([query_pts_per_batch, 1, 1])

    query_pt_feats, _ = torch.max(
        model["mlp"](torch.cat((query_embedding, pixel_feats, pointnet_feats), dim=2)),
        dim=1,
    )
    logits = model["occ_classifier"](query_pt_feats)[..., 0]

    """
    from mpl_toolkits.mplot3d import Axes3D
    f = figure()
    f.add_subplot(121, projection='3d')
    q = query_occ.bool().cpu().numpy()
    plot(sfm_xyz[:, 0], sfm_xyz[:, 1], sfm_xyz[:, 2], 'k.', markersize=0.1)
    plot(query_xyz[q, 0], query_xyz[q, 1], query_xyz[q, 2], 'r.', markersize=1)
    plot(query_xyz[~q, 0], query_xyz[~q, 1], query_xyz[~q, 2], 'b.', markersize=1)
    plot([anchor_pt.xyz[0]], [anchor_pt.xyz[1]], [anchor_pt.xyz[2]], 'g.')

    n = int(floor(len(uv_t) ** .5))
    figure()
    for i in range(n ** 2):
        subplot(n, n, i + 1)
        imshow(imgs[visible_img_inds[i]])
        plot(uv[i, 0], uv[i, 1], 'r.')
        axis('off')
    tight_layout()

    figure()
    for i in range(n ** 2):
        subplot(n, n, i + 1)
        plot(uv_t.cpu()[i, 0], uv_t.cpu()[i, 1], 'r.')
        axis('off')
        xlim((0, img_feats.shape[3]))
        ylim((0, img_feats.shape[2]))
    tight_layout()

    from mpl_toolkits.mplot3d import Axes3D
    f = figure()
    f.add_subplot(121, projection='3d')
    plot(sfm_xyz[:, 0], sfm_xyz[:, 1], sfm_xyz[:, 2], '.', markersize=1)
    plot(poses[:, 0, 3], poses[:, 1, 3], poses[:, 2, 3], 'b.')
    plot([anchor_pt.xyz[0]], [anchor_pt.xyz[1]], [anchor_pt.xyz[2]], 'r.')

    sfm_xyz_cam = (
        np.linalg.inv(poses) @ np.c_[sfm_xyz, np.ones(len(sfm_xyz))].T
    ).transpose(2, 0, 1)

    f.add_subplot(122, projection='3d')
    plot(sfm_xyz_cam[:, 0, 0], sfm_xyz_cam[:, 0, 1], sfm_xyz_cam[:, 0, 2], '.', markersize=1)
    plot([anchor_xyz_cam[0, 0]], [anchor_xyz_cam[0, 1]], [anchor_xyz_cam[0, 2]], 'r.')
    plot([0], [0], [0], 'b.')
    """

    # query_embedding = model['query_embedder'](rel_query_xyz)
    # pointnet_input = torch.cat((pixel_feats, query_embedding), axis=2)
    # pointnet_input = torch.cat((pixel_feats, rel_query_xyz), axis=2)
    # logits = model['pointnet'](pointnet_input)[:, :, 0]
    preds = torch.sigmoid(logits)

    # query_occ = query_occ[:, None].repeat(1, imgs_per_anchor)

    pos_inds = query_occ.bool()
    neg_inds = ~pos_inds
    pos_acc = torch.sum((preds > 0.5) & pos_inds) / pos_inds.sum().float()
    neg_acc = torch.sum((preds < 0.5) & neg_inds) / neg_inds.sum().float()

    true_pos = torch.sum((preds > 0.5) & pos_inds).float()
    all_predicted_pos = torch.sum(preds > 0.5)
    all_actual_pos = torch.sum(pos_inds)
    precision = true_pos / all_predicted_pos
    recall = true_pos / all_actual_pos

    occ_loss = occ_loss_fn(preds, query_occ)
    # cat_loss = cat_loss_fn(cat_logits, cat_gt[visible_img_inds])

    # loss = occ_loss + cat_loss
    loss = occ_loss

    loss.backward()
    optimizer.step()

    if _wandb:
        wandb.log(
            {
                "loss": occ_loss.item(),
                "cat_loss": cat_loss.item(),
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

torch.save(model, "bestmodel")
