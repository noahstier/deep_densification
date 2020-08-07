import functools
import glob
import importlib
import os

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


mlp = torch.nn.Sequential(
    FCLayer(160, 64), FCLayer(64), FCLayer(64), FCLayer(64), FCLayer(64), FCLayer(64),
)
classifier = torch.nn.Sequential(
    FCLayer(64, 64), FCLayer(64, 32), FCLayer(32, 8), torch.nn.Linear(8, 1)
)
pointnet_mlp = torch.nn.Sequential(
    FCLayer(3, 64), FCLayer(64), FCLayer(64), FCLayer(64), FCLayer(64), FCLayer(64),
)


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


_wandb = True
if _wandb:
    wandb.init(project="deepmvs")


house_dir = "/home/noah/Documents/scenecompletion/onetpp/data_preprocessing/suncg/output/000514ade3bcc292a613a4c2755a5050"

img_dir = os.path.join(house_dir, "imgs/color")
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

pil_imgs = [PIL.Image.open(os.path.join(img_dir, ims[i].name)) for i in im_ids]
imgs = np.stack([np.asarray(img) for img in pil_imgs], axis=0)
imgs_t = torch.stack([transform(img) for img in pil_imgs], dim=0).cuda()
imheight, imwidth, _ = imgs[0].shape

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
batch_size = 128
imgs_per_anchor = 6

model = torch.nn.ModuleDict(
    {
        # 'cnn': torchvision.models.mobilenet_v2(pretrained=True).features,
        "cnn": unet.UNet(3, 32),
        "mlp": mlp,
        "pointnet_mlp": pointnet_mlp,
        "classifier": classifier,
        "query_embedder": torch.nn.Sequential(
            FCLayer(3, 16), FCLayer(16, 32), FCLayer(32, 64)
        ),
    }
).cuda()

model.train()

if _wandb:
    wandb.watch(model)

feat_height, feat_width = model["cnn"](imgs_t[:1]).shape[2:]

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# {"lr": 1e-3, "params": model['cnn'].parameters()},
# {"lr": 1e-3, "params": model['mlp'].parameters()},
# {"lr": 1e-3, "params": model['classifier'].parameters()},
# {"lr": 1e-3, "params": model['query_embedder'].parameters()}
# ])

# pos_weight = torch.Tensor([np.mean(query_tsdf > 0) / np.mean(query_tsdf < 0)])
# loss_fn = torch.nn.BCELossWithLogits(pos_weight=pos_weight).cuda()
loss_fn = torch.nn.BCELoss().cuda()

for step in tqdm.trange(3000):
    optimizer.zero_grad()

    while True:
        index = np.random.randint(len(pt_ids))
        pt_id = pt_ids[index]
        anchor_pt = pts[pt_id]

        if len(anchor_pt.image_ids) < imgs_per_anchor:
            continue

        query_pt_inds = np.argwhere(
            np.linalg.norm(query_pts - anchor_pt.xyz, axis=1)
            < query_neighborhood_radius
        ).flatten()

        if len(query_pt_inds) < batch_size:
            continue

        query_pt_inds = np.random.choice(query_pt_inds, size=batch_size, replace=False)
        query_xyz = query_pts[query_pt_inds]
        query_occ = query_tsdf[query_pt_inds] < 0.001

        m = np.mean(query_occ)
        if m < 0.2 or m > 0.8:
            continue

        _, local_sfm_inds = kdtree.query(query_xyz, k=256)

        break
    query_occ = torch.Tensor(query_occ).cuda()

    visible_img_ids = anchor_pt.image_ids
    visible_img_inds = np.array([im_ids.index(i) for i in visible_img_ids])
    uv = np.stack(
        [
            ims[im_id].xys[point2D_idx]
            for im_id, point2D_idx in zip(anchor_pt.image_ids, anchor_pt.point2D_idxs)
        ],
        axis=0,
    )
    uv_t = uv / [imgs.shape[2], imgs.shape[1]] * [feat_width, feat_height]
    uv_t = torch.Tensor(uv_t).cuda()

    if len(visible_img_ids) > imgs_per_anchor:
        i = np.arange(len(visible_img_ids), dtype=int)
        i = np.random.choice(i, size=imgs_per_anchor, replace=False)
        visible_img_ids = visible_img_ids[i]
        visible_img_inds = visible_img_inds[i]
        uv = uv[i]
        uv_t = uv_t[i]

    poses = camera_extrinsics[visible_img_inds]

    query_xyz_cam = (
        np.linalg.inv(poses) @ np.c_[query_xyz, np.ones(len(query_xyz))].T
    ).transpose(2, 0, 1)
    anchor_xyz_cam = np.linalg.inv(poses) @ [*anchor_pt.xyz, 1]
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
    pixel_feats = interp_maps(img_feats, uv_t)
    pixel_feats = pixel_feats[None].repeat([batch_size, 1, 1])

    query_pt_feats, _ = torch.max(
        model["mlp"](torch.cat((query_embedding, pixel_feats, pointnet_feats), dim=2)),
        dim=1,
    )
    logits = model["classifier"](query_pt_feats)[..., 0]

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

    loss = loss_fn(preds, query_occ)

    loss.backward()
    optimizer.step()

    if _wandb:
        wandb.log(
            {
                "loss": loss.item(),
                "logits": wandb.Histogram(logits.detach().cpu().numpy()),
                "preds": wandb.Histogram(preds.detach().cpu().numpy()),
                "occ": wandb.Histogram(query_occ.cpu().numpy()),
                "pos_acc": pos_acc.item(),
                "neg_acc": neg_acc.item(),
            },
            step=step,
        )
