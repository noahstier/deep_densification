import functools
import glob
import importlib
import itertools
import os
import pickle

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


def interp_img(img, xy):
    x = xy[:, 0]
    y = xy[:, 1]

    x0 = torch.clamp(torch.floor(x).long(), 0, img.shape[2] - 1)
    y0 = torch.clamp(torch.floor(y).long(), 0, img.shape[1] - 1)

    x1 = torch.clamp(x0 + 1, 0, img.shape[2] - 1)
    y1 = torch.clamp(y0 + 1, 0, img.shape[1] - 1)

    f_ll = img[:, y0, x0]
    f_lr = img[:, y0, x1]
    f_ul = img[:, y1, x0]
    f_ur = img[:, y1, x1]

    interped = (
        f_ll * ((x - x0) * (y - y0))
        + f_lr * ((x1 - x) * (y - y0))
        + f_ur * ((x1 - x) * (y1 - y))
        + f_ul * ((x - x0) * (y1 - y))
    )
    return interped


def positional_encoding(xyz, L):
    encoding = []
    for l in range(L):
        encoding.append(np.sin(2 ** l ** np.pi * xyz))
        encoding.append(np.cos(2 ** l ** np.pi * xyz))
    encoding = np.concatenate(encoding, axis=-1)
    return encoding


def pts_in_frustum(pts, frustum_pts):
    triangles = np.stack(
        [
            frustum_pts[[4, 0, 3]],
            frustum_pts[[4, 2, 1]],
            frustum_pts[[4, 3, 2]],
            frustum_pts[[4, 1, 0]],
            frustum_pts[[1, 2, 3]],
        ],
        axis=0,
    )
    pt_to_vert = pts[:, None] - triangles[:, 0][None]

    vert2_to_vert = triangles[:, 2] - triangles[:, 0]
    vert1_to_vert = triangles[:, 1] - triangles[:, 0]

    inward_vectors = np.cross(vert2_to_vert, vert1_to_vert, axis=-1)
    dotprods = np.sum(pt_to_vert * inward_vectors[None], axis=-1)
    inside = np.all(dotprods > 0, axis=-1)
    return inside


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

model = torch.nn.ModuleDict(
    {
        "mlp": torch.nn.Sequential(
            FCLayer(35, 256),
            FCLayer(256),
            FCLayer(256),
            FCLayer(256),
            FCLayer(256),
            FCLayer(256),
            FCLayer(256),
            FCLayer(256, 64),
            FCLayer(64, 16),
            torch.nn.Linear(16, 1),
        ),
        "mini_mlp": torch.nn.Sequential(
            FCLayer(1280, 512),
            FCLayer(512, 256),
            FCLayer(256, 128),
            FCLayer(128, 64),
            FCLayer(64, 32),
        ),
        # 'cnn': torchvision.models.mobilenet_v2(pretrained=True).features[:7],
        "cnn": torchvision.models.mobilenet_v2(pretrained=True).features,
    }
).cuda()
# featheight, featwidth = model['cnn'](imgs_t[:1]).shape[2:]


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

print("pre-computing frustum pts")
"""
per_img_query_pts = {}
img_corners_img = np.array(
    [[0, 0], [imwidth, 0], [imwidth, imheight], [0, imheight],]
)
img_corners_cam = (
    np.linalg.inv(camera_intrinsic) @ np.c_[img_corners_img, np.ones(4)].T
).T
maxrange = 100
for i, im_id in enumerate(tqdm.tqdm(im_ids)):
    camera_world = camera_extrinsics[i, :3, 3]
    img_corners_world = (camera_extrinsics[i] @ np.c_[img_corners_cam, np.ones(4)].T).T[:, :3]
    img_corners_world = (img_corners_world - camera_world) * maxrange + camera_world
    frustum_pts = np.c_[img_corners_world.T, camera_world].T
    inside = pts_in_frustum(query_pts, frustum_pts)

    query_xyz = query_pts[inside]
    query_xyz_cam = (np.linalg.inv(camera_extrinsics[i]) @ np.c_[query_xyz, np.ones(len(query_xyz))].T).T[:, :3]

    query_uv = (camera_intrinsic @ query_xyz_cam.T).T
    query_uv = query_uv[:, :2] / query_uv[:, 2:]
    # query_uv = np.clip(query_uv, [0, 0], [imwidth - 1, imheight - 1])

    query_uv_t = query_uv / [imwidth, imheight] * [featwidth, featheight]

    normed_xyz_cam = query_xyz_cam - np.array([-3.47185991, -2.54137404,  0.022674  ])
    normed_xyz_cam /= np.array([6.63462202, 3.68160095, 6.79344803])
    normed_xyz_cam = normed_xyz_cam * 2 - 1

    # encoded_xyz_cam = positional_encoding(normed_xyz_cam, L=1)
    encoded_xyz_cam = normed_xyz_cam

    per_img_query_pts[im_id] = {
        'xyz': query_xyz,
        'xyz_cam': query_xyz_cam,
        'encoded_xyz_cam': encoded_xyz_cam,
        'occ': query_tsdf[inside] < 0,
        'uv': query_uv,
        'uv_t': query_uv_t,
    }

import pickle
with open('pt_data.pkl', 'wb') as f:
    pickle.dump(per_img_query_pts, f)

import sys
sys.exit(0)
"""


with open("pt_data.pkl", "rb") as f:
    per_img_query_pts = pickle.load(f)

model.train()

if config.wandb:
    wandb.watch(model)

optimizer = torch.optim.Adam(model.parameters())

# pos_weight = torch.Tensor([np.mean(query_tsdf > 0) / np.mean(query_tsdf < 0)])
# occ_loss_fn = torch.nn.BCELossWithLogits(pos_weight=pos_weight).cuda()
occ_loss_fn = torch.nn.BCELoss().cuda()
cat_loss_fn = torch.nn.CrossEntropyLoss().cuda()

step = -1
for epoch in range(20_000):
    print("epoch {}".format(epoch))
    indices = np.arange(len(imgs), dtype=int)
    np.random.shuffle(indices)
    indices = iter(indices)

    for _ in tqdm.trange(len(imgs) // config.img_batch_size):
        optimizer.zero_grad()

        img_inds = np.array([next(indices) for _ in range(config.img_batch_size)])
        img_feats = torch.relu(model["cnn"](imgs_t[img_inds]))

        img_feats = torch.mean(img_feats, dim=2)
        img_feats = torch.mean(img_feats, dim=2)

        query_coords = []
        query_xyz_world = []
        query_occ = []
        # pixel_feats = []
        for i, ind in enumerate(img_inds):
            im_id = im_ids[ind]
            inds = np.arange(
                len(per_img_query_pts[im_id]["encoded_xyz_cam"]), dtype=int
            )
            inds = np.random.choice(inds, size=config.pt_batch_size, replace=False)

            """
            j = 0

            query_uv = per_img_query_pts[im_id]['uv'][inds]
            occ = per_img_query_pts[im_id]['occ'][inds][j]

            subplot(131)
            imshow(imgs[ind])
            plot(query_uv[j, 0], query_uv[j, 1], 'b.')
            axis("off")

            gcf().add_subplot(132, projection='3d')
            query_xyz = per_img_query_pts[im_id]['xyz'][inds[j]]
            plot(sfm_xyz[:, 0], sfm_xyz[:, 1], sfm_xyz[:, 2], 'k.', markersize=1)
            # plot(per_img_query_pts[im_id]['xyz'][:, 0], per_img_query_pts[im_id]['xyz'][:, 1], per_img_query_pts[im_id]['xyz'][:, 2], '.')
            plot([query_xyz[0]], [query_xyz[1]], [query_xyz[2]], 'b.')

            near_sfm_inds = np.linalg.norm(sfm_xyz - query_xyz, axis=-1) < .3
            near_sfm_xyz = sfm_xyz[near_sfm_inds]
            near_sfm_rgb = sfm_rgb[near_sfm_inds]
            gcf().add_subplot(133, projection='3d')
            gca().scatter(near_sfm_xyz[:, 0], near_sfm_xyz[:, 1], near_sfm_xyz[:, 2], c=near_sfm_rgb.astype(np.float32) / 255)
            plot([query_xyz[0]], [query_xyz[1]], [query_xyz[2]], 'b.' if occ else 'r.')

            tight_layout()

            query_xyz = per_img_query_pts[im_id]['xyz']
            occ = per_img_query_pts[im_id]['occ']
            colors = np.array([[0, 0, 1], [1, 0, 0]], dtype=np.float32)[occ.astype(int)]
            gcf().add_subplot(111, projection='3d')
            gca().scatter(sfm_xyz[:, 0], sfm_xyz[:, 1], sfm_xyz[:, 2], s=.1, c='k')
            gca().scatter(query_xyz[:, 0], query_xyz[:, 1], query_xyz[:, 2], c=colors)

            """

            query_uv_t = torch.Tensor(per_img_query_pts[im_id]["uv_t"][inds]).cuda()
            # pixel_feats.append(interp_img(img_feats[i], query_uv_t).T)
            query_coords.append(per_img_query_pts[im_id]["encoded_xyz_cam"][inds])
            query_occ.append(per_img_query_pts[im_id]["occ"][inds])
            query_xyz_world.append(per_img_query_pts[im_id]["xyz"][inds])

        query_coords = torch.Tensor(np.concatenate(query_coords, axis=0)).cuda()
        query_occ = torch.Tensor(np.concatenate(query_occ, axis=0)).cuda()
        # pixel_feats = torch.cat(pixel_feats, dim=0)

        # query_xyz_world = np.concatenate(query_xyz_world, axis=0)
        # query_xyz_world -= np.array([3.75210289e+01, 4.46640313e-02, 4.68986430e+01])
        # query_xyz_world /= np.array([7.52712135, 2.74440703, 6.09322482])
        # query_xyz_world = query_xyz_world * 2 - 1
        # query_xyz_world = torch.Tensor(query_xyz_world).cuda()

        # mlp_input = torch.cat((query_coords, pixel_feats), dim=1)
        # logits = model['mlp'](mlp_input)[..., 0]
        # img_ind_encoding = np.array([img_inds / len(imgs) * 2 - 1]).T
        # img_ind_encoding = positional_encoding(img_ind_encoding, L=2)
        # img_ind_encoding = np.tile(img_ind_encoding, (1, config.pt_batch_size)).reshape(config.pt_batch_size * config.img_batch_size, -1)
        # img_ind_encoding = torch.Tensor(img_ind_encoding).cuda()

        img_feats = model["mini_mlp"](img_feats)
        img_feats = img_feats.repeat((1, config.pt_batch_size)).reshape(
            config.pt_batch_size * config.img_batch_size, -1
        )

        mlp_input = torch.cat((query_coords, img_feats), dim=1)
        logits = model["mlp"](mlp_input)[..., 0]

        preds = torch.sigmoid(logits)

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

    name = "im-cond"
    torch.save(model.state_dict(), os.path.join("models", name))
