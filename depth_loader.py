import glob
import os

import config

import cv2
import numba
import numpy as np
import PIL.Image
import scipy.spatial
import torch
import torchvision
import tqdm

imwidth = 640
imheight = 480

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


@numba.jit(nopython=True)
def take_first_dists_below_thresh(a, b, n, thresh):
    result = np.ones((a.shape[0], n), dtype=np.int32) * -1
    for i in range(a.shape[0]):
        k = 0
        a_val = a[i]
        for j in range(b.shape[0]):
            b_val = b[j]
            dist = np.sum((a_val - b_val) ** 2) ** 0.5
            if dist < thresh:
                result[i, k] = j
                k += 1
                if k > n:
                    break
    return result


def positional_encoding(xyz, L):
    encoding = []
    for l in range(L):
        encoding.append(np.sin(2 ** l ** np.pi * xyz))
        encoding.append(np.cos(2 ** l ** np.pi * xyz))
    encoding = np.concatenate(encoding, axis=-1)
    return encoding


class Dataset(torch.utils.data.Dataset):
    def __init__(self, house_dirs, n_anchors, n_queries_per_anchor):
        self.n_anchors = n_anchors
        self.n_queries_per_anchor = n_queries_per_anchor
        self.images = []

        print("initializing dataset")
        for house_dir in tqdm.tqdm(house_dirs):
            rgb_imgdir = os.path.join(house_dir, "imgs/color")
            depth_imgdir = os.path.join(house_dir, "imgs/depth")
            rgb_imgfiles = sorted(glob.glob(os.path.join(rgb_imgdir, "*")))
            depth_imgfiles = sorted(glob.glob(os.path.join(depth_imgdir, "*")))
            for img_ind, (rgb_imgfile, depth_imgfile) in enumerate(
                zip(rgb_imgfiles, depth_imgfiles)
            ):
                self.images.append((house_dir, img_ind, rgb_imgfile, depth_imgfile))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        house_dir, img_ind, rgb_imgfile, depth_imgfile = self.images[index]

        fusion_npz = np.load(os.path.join(house_dir, "fusion.npz"))
        sfm_imfile = os.path.join(house_dir, "sfm/sparse/auto/images.bin")
        sfm_ptfile = os.path.join(house_dir, "sfm/sparse/auto/points3D.bin")

        pose_npz = np.load(os.path.join(house_dir, "poses.npz"))

        depth_img = (
            cv2.imread(depth_imgfile, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000
        )

        intrinsic = pose_npz["intrinsic"]
        extrinsic = pose_npz["extrinsics"][img_ind]

        query_xyz = fusion_npz["query_pts"]
        query_tsdf = fusion_npz["query_tsdf"]

        query_xyz_cam = (
            np.linalg.inv(extrinsic) @ np.c_[query_xyz, np.ones(len(query_xyz))].T
        ).T[:, :3]
        query_uv = query_xyz_cam @ intrinsic.T
        query_uv = query_uv[:, :2] / query_uv[:, 2:]

        inds = (
            (query_xyz_cam[:, 2] > 0)
            & (query_uv[:, 0] >= 0)
            & (query_uv[:, 1] >= 0)
            & (query_uv[:, 0] < imwidth)
            & (query_uv[:, 1] < imheight)
        )

        inds = np.argwhere(inds).flatten()
        np.random.shuffle(inds)

        anchor_inds = np.random.choice(inds, size=self.n_anchors * 3, replace=False)
        anchor_uv = query_uv[anchor_inds]
        anchor_uv_inds = np.floor(anchor_uv).astype(int)

        query_xyz = query_xyz[inds]
        query_xyz_cam = query_xyz_cam[inds]
        query_uv = query_uv[inds]
        query_tsdf = query_tsdf[inds]

        anchor_ranges = depth_img[anchor_uv_inds[:, 1], anchor_uv_inds[:, 0]]
        anchor_xyz_cam = (
            np.linalg.inv(intrinsic) @ np.c_[anchor_uv, np.ones(len(anchor_uv))].T
        ).T * anchor_ranges[:, None]

        query_inds = take_first_dists_below_thresh(
            anchor_xyz_cam, query_xyz_cam, self.n_queries_per_anchor, 0.2
        )

        good_anchor_inds = ~np.any(query_inds == -1, axis=1)
        if np.sum(good_anchor_inds) < self.n_anchors:
            return self[np.random.randint(0, len(self))]

        selected_anchor_inds = np.random.choice(
            np.argwhere(good_anchor_inds).flatten(), size=self.n_anchors, replace=False
        )

        anchor_uv = anchor_uv[selected_anchor_inds]
        anchor_ranges = anchor_ranges[selected_anchor_inds]
        anchor_xyz_cam = anchor_xyz_cam[selected_anchor_inds]
        query_inds = query_inds[selected_anchor_inds]

        query_xyz = query_xyz[query_inds]
        query_xyz_cam = query_xyz_cam[query_inds]
        query_uv = query_uv[query_inds]
        query_tsdf = query_tsdf[query_inds]

        anchor_cam_unit = anchor_xyz_cam / np.linalg.norm(
            anchor_xyz_cam, axis=-1, keepdims=True
        )
        cross = np.c_[
            -anchor_xyz_cam[:, 1], anchor_xyz_cam[:, 0], np.zeros(len(anchor_xyz_cam))
        ]
        cross /= np.linalg.norm(cross, axis=-1, keepdims=True)
        dot = anchor_cam_unit[:, 2]
        axis = cross
        if np.any(np.isnan(axis)):
            return self[np.random.randint(0, len(self))]
        angle = np.arccos(dot)

        angle = np.random.uniform(0, 2 * np.pi, size=angle.shape)
        axis = np.c_[np.random.uniform(-1, 1, size=(len(axis), 2)), np.zeros(len(axis))]
        axis /= np.linalg.norm(axis, axis=-1, keepdims=True)

        cam2anchor_rot = scipy.spatial.transform.Rotation.from_rotvec(
            axis * angle[:, None]
        ).as_matrix()

        anchor_xyz_rotated = np.stack(
            [
                (np.linalg.inv(cam2anchor_rot[i]) @ anchor_xyz_cam[i].T).T
                for i in range(len(anchor_xyz_cam))
            ],
            axis=0,
        )

        query_xyz_rotated = np.stack(
            [
                (np.linalg.inv(cam2anchor_rot[i]) @ query_xyz_cam[i].T).T
                for i in range(len(anchor_xyz_cam))
            ],
            axis=0,
        )

        rel_query_xyz = query_xyz_rotated - anchor_xyz_rotated[:, None]

        query_coords = positional_encoding(
            (query_xyz_rotated - anchor_xyz_rotated[:, None]) / 0.2, L=1
        )
        query_occ = query_tsdf < 0

        pil_img = PIL.Image.open(rgb_imgfile)
        rgb_img = np.asarray(pil_img)
        rgb_img_t = transform(pil_img)

        query_uv_t = (
            query_uv / [imwidth, imheight] * [rgb_img_t.shape[2], rgb_img_t.shape[1]]
        )

        anchor_uv_t = (
            anchor_uv / [imwidth, imheight] * [rgb_img_t.shape[2], rgb_img_t.shape[1]]
        )

        return (
            query_coords.astype(np.float32),
            query_occ,
            query_uv,
            query_uv_t,
            query_xyz,
            query_xyz_cam,
            anchor_uv,
            anchor_uv_t,
            anchor_xyz_cam,
            rgb_img_t,
            rgb_img,
            extrinsic,
            intrinsic,
        )


if __name__ == "__main__":
    import importlib

    spec = importlib.util.spec_from_file_location(
        "colmap_reader", config.colmap_reader_script
    )
    colmap_reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(colmap_reader)

    house_dirs = sorted([d for d in glob.glob(os.path.join(config.dset_dir, "*"))])

    """
    house_dir = np.random.choice(house_dirs)
    npzfile = os.path.join(house_dir, 'fusion.npz')
    npz = np.load(npzfile)
    tsdf = npz['tsdf_point_cloud']
    reproj_pts = npz['reprojection_samples']
    reproj_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(reproj_pts))
    minbounds = np.percentile(reproj_pts, 2, axis=0)
    maxbounds = np.percentile(reproj_pts, 98, axis=0)
    maxbounds += .1
    minbounds -= .1
    room_corners = [
        [minbounds[0], minbounds[1], minbounds[2]],
        [maxbounds[0], minbounds[1], minbounds[2]],
        [minbounds[0], minbounds[1], maxbounds[2]],
        [maxbounds[0], minbounds[1], maxbounds[2]],
        [minbounds[0], maxbounds[1], minbounds[2]],
        [maxbounds[0], maxbounds[1], minbounds[2]],
        [minbounds[0], maxbounds[1], maxbounds[2]],
        [maxbounds[0], maxbounds[1], maxbounds[2]],
    ]
    bounds_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate((
        np.linspace(room_corners[0], room_corners[1], 100),
        np.linspace(room_corners[0], room_corners[2], 100),
        np.linspace(room_corners[1], room_corners[3], 100),
        np.linspace(room_corners[2], room_corners[3], 100),
        np.linspace(room_corners[4], room_corners[5], 100),
        np.linspace(room_corners[4], room_corners[6], 100),
        np.linspace(room_corners[5], room_corners[7], 100),
        np.linspace(room_corners[6], room_corners[7], 100),
        np.linspace(room_corners[0], room_corners[4], 100),
        np.linspace(room_corners[1], room_corners[5], 100),
        np.linspace(room_corners[2], room_corners[6], 100),
        np.linspace(room_corners[3], room_corners[7], 100),
    ), axis=0)))
    bounds_pcd.paint_uniform_color(np.array([0, 1, 0], dtype=np.float64))
    query_xyz = npz['query_pts']
    query_occ = npz['query_tsdf'] < 0
    query_rgb = np.array([[1, 0, 0], [0, 0, 1]])[query_occ.astype(int)]
    query_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(query_xyz))
    query_pcd.colors = o3d.utility.Vector3dVector(query_rgb)
    gt_xyz = tsdf[:, :3]
    gt_rgb = tsdf[:, 3:] / 255
    gt_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt_xyz))
    gt_pcd.colors = o3d.utility.Vector3dVector(gt_rgb)
    # o3d.visualization.draw_geometries([query_pcd, gt_pcd])
    o3d.visualization.draw_geometries([query_pcd, reproj_pcd, gt_pcd, bounds_pcd])
    """

    dset = Dataset(house_dirs, n_anchors=16, n_queries_per_anchor=128)
    index = 0
    self = dset

    (
        query_coords,
        query_occ,
        query_uv,
        query_uv_t,
        query_xyz,
        query_xyz_cam,
        rgb_img_t,
        rgb_img,
        extrinsic,
        intrinsic,
    ) = dset[index]

    house_dir = dset.images[index][0]

    sfm_ptfile = os.path.join(house_dir, "sfm/sparse/auto/points3D.bin")
    pts = colmap_reader.read_points3d_binary(sfm_ptfile)
    pts = {
        pt_id: pt
        for pt_id, pt in pts.items()
        if pt.error < 1 and len(pt.image_ids) >= 5
    }
    pt_ids = sorted(pts.keys())
    sfm_xyz = np.stack([pts[i].xyz for i in pt_ids], axis=0)
    sfm_rgb = np.stack([pts[i].rgb for i in pt_ids], axis=0)

    sfm_xyz_cam = (
        np.linalg.inv(extrinsic) @ np.c_[sfm_xyz, np.ones(len(sfm_xyz))].T
    ).T[:, :3]
    sfm_uv = (intrinsic @ sfm_xyz_cam.T).T
    sfm_uv = sfm_uv[:, :2] / sfm_uv[:, 2:]
    inds = (
        (sfm_xyz_cam[:, 2] > 0)
        & (sfm_uv[:, 0] >= 0)
        & (sfm_uv[:, 1] >= 0)
        & (sfm_uv[:, 0] < imwidth)
        & (sfm_uv[:, 1] < imheight)
    )

    subplot(131)
    imshow(rgb_img)
    scatter(
        query_uv[:, 0],
        query_uv[:, 1],
        s=1,
        c=plt.cm.jet(np.linspace(0, 1, len(query_uv))),
    )
    axis("off")

    gcf().add_subplot(132, projection="3d")
    gca().scatter(
        sfm_xyz[:, 0],
        sfm_xyz[:, 1],
        sfm_xyz[:, 2],
        s=1,
        c=sfm_rgb.astype(np.float32) / 255,
    )
    gca().scatter(
        query_xyz[:, 0],
        query_xyz[:, 1],
        query_xyz[:, 2],
        s=1,
        c=np.array([[1, 0, 0], [0, 0, 1]])[query_occ.astype(int)],
    )
    plot([extrinsic[0, 3]], [extrinsic[1, 3]], [extrinsic[2, 3]], "g.")

    gcf().add_subplot(133, projection="3d")
    gca().scatter(
        sfm_xyz_cam[inds, 0],
        sfm_xyz_cam[inds, 1],
        sfm_xyz_cam[inds, 2],
        s=1,
        c=sfm_rgb[inds].astype(np.float32) / 255,
    )
    gca().scatter(
        query_xyz_cam[:, 0],
        query_xyz_cam[:, 1],
        query_xyz_cam[:, 2],
        s=1,
        c=np.array([[1, 0, 0], [0, 0, 1]])[query_occ.astype(int)],
    )
    plot([0], [0], [0], "g.")

    tight_layout()
