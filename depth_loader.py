import glob
import os

import config

import cv2
import numba
import numpy as np
import PIL.Image
import scipy.spatial
import open3d as o3d
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
def take_first_dists_below_thresh3(a, b, n, thresh):
    result = np.ones((a.shape[0], n), dtype=np.int32) * -1
    for i in range(a.shape[0]):
        k = 0
        a_val = a[i]
        for j in range(b.shape[0]):
            b_val = b[j]
            dist = (
                (a_val[0] - b_val[0]) ** 2
                + (a_val[1] - b_val[1]) ** 2
                + (a_val[2] - b_val[2]) ** 2
            ) ** 0.5
            if dist < thresh:
                result[i, k] = j
                k += 1
                if k == n:
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
                """
                """
                cat_imgfile = rgb_imgfile.replace("color", "category").replace(
                    "jpg", "png"
                )
                if (
                    np.sum(np.asarray(PIL.Image.open(cat_imgfile)) == 11)
                    < self.n_anchors * 3
                ):
                    continue
                """
                """
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

        # narrow down query points to within frustum, and shuffle them

        inds = (
            (query_xyz_cam[:, 2] > 0)
            & (query_uv[:, 0] >= 0)
            & (query_uv[:, 1] >= 0)
            & (query_uv[:, 0] < imwidth)
            & (query_uv[:, 1] < imheight)
        )

        inds = np.argwhere(inds).flatten()
        np.random.shuffle(inds)

        """
        """
        cat_imgfile = rgb_imgfile.replace("color", "category").replace("jpg", "png")
        cat_img = np.asarray(
            PIL.Image.open(cat_imgfile).transpose(PIL.Image.FLIP_TOP_BOTTOM)
        )
        anchor_uv = anchor_uv_inds = np.argwhere(cat_img == 11)[:, [1, 0]]
        __inds = np.random.choice(
            np.arange(len(anchor_uv)), size=self.n_anchors * 3, replace=False
        )
        anchor_uv = anchor_uv[__inds]
        anchor_uv_inds = anchor_uv_inds[__inds]

        """
        """

        """
        # pick anchors at pixels where there is already a query point --
        # better chance that there will be enough query points around this anchor
        anchor_inds = np.random.choice(inds, size=self.n_anchors * 3, replace=False)
        anchor_uv = query_uv[anchor_inds]
        anchor_uv_inds = np.floor(anchor_uv).astype(int)
        """

        query_xyz = query_xyz[inds]
        query_xyz_cam = query_xyz_cam[inds]
        query_uv = query_uv[inds]
        query_tsdf = query_tsdf[inds]

        anchor_ranges = depth_img[anchor_uv_inds[:, 1], anchor_uv_inds[:, 0]]
        anchor_xyz_cam = (
            np.linalg.inv(intrinsic) @ np.c_[anchor_uv, np.ones(len(anchor_uv))].T
        ).T * anchor_ranges[:, None]

        """
        a = anchor_xyz_cam
        b = query_xyz_cam
        n = self.n_queries_per_anchor
        thresh = 0.2
        """

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
        center_pixel_u = np.array([0, 0, 1])
        cross = np.cross(center_pixel_u, anchor_cam_unit)
        cross /= np.linalg.norm(cross, axis=-1, keepdims=True)
        dot = np.dot(center_pixel_u, anchor_cam_unit.T)
        axis = cross
        if np.any(np.isnan(axis)):
            return self[np.random.randint(0, len(self))]
        angle = np.arccos(dot)

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

        query_coords = (query_xyz_rotated - anchor_xyz_rotated[:, None]) / 0.2
        # query_coords = positional_encoding(
        #     (query_xyz_rotated - anchor_xyz_rotated[:, None]) / 0.2, L=1
        # )
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

        """
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

        sfm_xyz_cam_rotated = np.stack(
            [
                (np.linalg.inv(cam2anchor_rot[i]) @ sfm_xyz_cam.T).T
                for i in range(len(anchor_xyz_cam))
            ],
            axis=0,
        )


        figure()
        subplot(121)
        imshow(rgb_img)
        scatter(sfm_uv[inds, 0], sfm_uv[inds, 1], c=sfm_rgb[inds].astype(np.float32) / 255)
        gcf().add_subplot(122, projection='3d')
        gca().scatter(
            sfm_xyz[inds, 0],
            sfm_xyz[inds, 1],
            sfm_xyz[inds, 2],
            c=sfm_rgb[inds].astype(np.float32) / 255
        )
        plot([extrinsic[0, 3]], [extrinsic[1, 3]], [extrinsic[2, 3]], 'b.')

        gcf().add_subplot(121, projection='3d')
        plot(sfm_xyz[inds, 0], sfm_xyz[inds, 1], sfm_xyz[inds, 2], '.') #, c=sfm_rgb[inds].astype(np.float32) / 255)
        plot([extrinsic[0, 3]], [extrinsic[1, 3]], [extrinsic[2, 3]], '.')
        subplot(122)
        imshow(rgb_img)
        xlabel('x')
        ylabel('y')

        gcf().add_subplot(121, projection='3d')
        plot(sfm_xyz_cam[inds, 0], sfm_xyz_cam[inds, 1], sfm_xyz_cam[inds, 2], '.') #, c=sfm_rgb[inds].astype(np.float32) / 255)
        plot([0], [0], [0], '.')
        xlabel('x')
        ylabel('y')
        subplot(122)
        imshow(rgb_img)

        qxr_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(query_xyz_rotated[j]))
        qxr_pcd.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0], [0, 0, 1]])[query_occ[j].astype(int)])

        sxr_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sfm_xyz_cam_rotated[j]))
        sxr_pcd.colors = o3d.utility.Vector3dVector(sfm_rgb.astype(float) / 255)
        o3d.visualization.draw_geometries([qxr_pcd, sxr_pcd])

        figure()
        gcf().add_subplot(121, projection='3d')
        gca().scatter(
            query_xyz_rotated[j, :, 0],
            query_xyz_rotated[j, :, 1],
            query_xyz_rotated[j, :, 2],
            s=1,
            c=np.array([[1, 0, 0], [0, 0, 1]])[query_occ[j].astype(int)],
        )
        plot(*[[i] for i in anchor_xyz_rotated[j]], '.')
        xlabel('x')
        ylabel('y')
        plot([0], [0], [0], 'g.')
        gca().scatter(
            sfm_xyz_cam_rotated[j, inds, 0],
            sfm_xyz_cam_rotated[j, inds, 1],
            sfm_xyz_cam_rotated[j, inds, 2],
            s=1,
            c=sfm_rgb[inds].astype(np.float32) / 255,
        )

        subplot(122)
        imshow(rgb_img)
        plot(query_uv[0, :, 0], query_uv[0, :, 1], '.')
        """

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
            depth_img,
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
    """
    with open("w_chair.txt", "r") as f:
        house_names_w_chair = set(f.read().split())
    house_dirs = [
        h for h in house_dirs[:10] if os.path.basename(h) in house_names_w_chair
    ]
    """
    """

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
        anchor_uv,
        anchor_uv_t,
        anchor_xyz_cam,
        rgb_img_t,
        rgb_img,
        depth_img,
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

    j = 0
    subplot(131)
    imshow(rgb_img)
    scatter(
        query_uv[j, :, 0],
        query_uv[j, :, 1],
        s=1,
        c=plt.cm.jet(np.linspace(0, 1, query_uv.shape[1])),
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
        query_xyz[j, :, 0],
        query_xyz[j, :, 1],
        query_xyz[j, :, 2],
        s=1,
        c=np.array([[1, 0, 0], [0, 0, 1]])[query_occ[j].astype(int)],
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
        query_xyz_cam[j, :, 0],
        query_xyz_cam[j, :, 1],
        query_xyz_cam[j, :, 2],
        s=1,
        c=np.array([[1, 0, 0], [0, 0, 1]])[query_occ[j].astype(int)],
    )
    plot([0], [0], [0], "g.")
    xlabel("x")
    ylabel("y")

    tight_layout()

    # check depth projection
    x = np.arange(depth_img.shape[1])
    y = np.arange(depth_img.shape[0])
    xx, yy = np.meshgrid(x, y)
    uv = np.c_[xx.flatten(), yy.flatten()]
    pixel_u = (np.linalg.inv(intrinsic) @ np.c_[uv, np.ones(len(uv))].T).T
    pixel_u /= np.linalg.norm(pixel_u, axis=-1, keepdims=True)
    pixel_ranges = depth_img[uv[:, 1], uv[:, 0]]
    xyz_cam = pixel_u * pixel_ranges[:, None]
    xyz = (extrinsic @ np.c_[xyz_cam, np.ones(len(xyz_cam))].T).T[:, :3]
    figure()
    subplot(121)
    imshow(depth_img)
    gcf().add_subplot(122, projection="3d")
    plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], ".", markersize=0.1)
    plot([extrinsic[0, 3]], [extrinsic[1, 3]], [extrinsic[2, 3]], ".")
    gca().scatter(
        sfm_xyz[inds, 0],
        sfm_xyz[inds, 1],
        sfm_xyz[inds, 2],
        c=sfm_rgb[inds].astype(np.float32) / 255,
    )

    # check query coords
    figure()
    gcf().add_subplot(111, projection="3d")
    plot([0], [0], [0], ".")
    gca().scatter(
        query_coords[j, :, 0],
        query_coords[j, :, 1],
        query_coords[j, :, 2],
        s=1,
        c=np.array([[1, 0, 0], [0, 0, 1]])[query_occ[j].astype(int)],
    )
    xlabel("x")
    ylabel("y")
