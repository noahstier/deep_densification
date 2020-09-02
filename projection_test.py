import functools
import glob
import importlib
import itertools
import os
import pickle

import cv2
import imageio
import numpy as np
import open3d as o3d
import pickle
import PIL.Image
import scipy.spatial
import scipy.ndimage
import skimage.measure
import torch
import torchvision
import tqdm
import wandb

import fpn
import config
import unet

torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(1)


spec = importlib.util.spec_from_file_location(
    "colmap_reader", config.colmap_reader_script
)
colmap_reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(colmap_reader)

house_dirs = sorted(
    [
        d
        for d in glob.glob(os.path.join(config.dset_dir, "*"))
        if os.path.exists(os.path.join(d, "fusion.npz"))
        and os.path.exists(os.path.join(d, "poses.npz"))
    ]
)
test_house_dirs = house_dirs[:10]
train_house_dirs = house_dirs[10:]

house_dir = test_house_dirs[-2]

rgb_imgdir = os.path.join(house_dir, "imgs/color")
depth_imgdir = os.path.join(house_dir, "imgs/depth")
cat_imgdir = os.path.join(house_dir, "imgs/category")
fusion_npz = np.load(os.path.join(house_dir, "fusion.npz"))
sfm_imfile = os.path.join(house_dir, "sfm/sparse/auto/images.bin")
sfm_ptfile = os.path.join(house_dir, "sfm/sparse/auto/points3D.bin")

ims = colmap_reader.read_images_binary(sfm_imfile)
pts = colmap_reader.read_points3d_binary(sfm_ptfile)

pts = {
    pt_id: pt for pt_id, pt in pts.items() if pt.error < 1 and len(pt.image_ids) >= 5
}

im_ids = sorted(ims.keys())
pt_ids = sorted(pts.keys())

pil_imgs = [PIL.Image.open(os.path.join(rgb_imgdir, ims[i].name)) for i in im_ids]
imgs = np.stack([np.asarray(img) for img in pil_imgs], axis=0)
imheight, imwidth, _ = imgs[0].shape


cat_imgs = np.stack(
    [
        PIL.Image.open(
            os.path.join(cat_imgdir, ims[im_id].name.replace("jpg", "png"))
        ).transpose(PIL.Image.FLIP_TOP_BOTTOM)
        for im_id in im_ids
    ],
    axis=0,
)

depth_imgs = (
    np.stack(
        [
            cv2.imread(
                os.path.join(depth_imgdir, ims[im_id].name.replace("jpg", "png")),
                cv2.IMREAD_ANYDEPTH,
            )
            for im_id in im_ids
        ],
        axis=0,
    ).astype(np.float32)
    / 1000
)

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

m = []
for i in range(len(pt_ids)):
    pt_id = pt_ids[i]
    pt = pts[pt_id]

    uv1 = np.stack(
        [ims[im_id].xys[idx] for im_id, idx in zip(pt.image_ids, pt.point2D_idxs)],
        axis=0,
    )

    img_inds = [im_ids.index(im_id) for im_id in pt.image_ids]
    xyz_cam = (np.linalg.inv(camera_extrinsics[img_inds]) @ [*pt.xyz, 1])[:, :3]
    uv2 = (camera_intrinsic @ xyz_cam.T).T
    uv2 = uv2[:, :2] / uv2[:, 2:]

    dists = np.linalg.norm(uv1 - uv2, axis=1)
    maxdist = np.max(dists)
    m.append(maxdist)


for i, im_id in enumerate(pt.image_ids):

    subplot(4, 4, i + 1)
    imshow(imgs[img_inds[i]])
    plot(uv1[i, 0], uv1[i, 1], "b.")
    plot(uv2[i, 0], uv2[i, 1], "r.")
    axis("off")

tight_layout()
