import os
import glob

import cv2
import imageio
import numpy as np
import open3d as o3d
import scipy.spatial
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, scan_dirs, n_imgs, split=None):
        super().__init__()
        self.scan_dirs = scan_dirs
        self.n_imgs = n_imgs

        if split not in ["train", "test"]:
            raise Exception()
        self.split = split

        self.rgb_imgfiles = []
        self.depth_imgfiles = []
        for scan_dir in self.scan_dirs:
            rgb_imgfiles = sorted(
                glob.glob(os.path.join(scan_dir, "color/*.jpg")),
                key=lambda f: int(os.path.basename(f).split(".")[0]),
            )
            depth_imgfiles = sorted(
                glob.glob(os.path.join(scan_dir, "depth/*.png")),
                key=lambda f: int(os.path.basename(f).split(".")[0]),
            )
            self.rgb_imgfiles.append(rgb_imgfiles)
            self.depth_imgfiles.append(depth_imgfiles)

    def __len__(self):
        return len(self.rgb_imgfiles)

    def __getitem__(self, index):
        scan_dir = self.scan_dirs[index]
        rgb_imgfiles = self.rgb_imgfiles[index]
        depth_imgfiles = self.depth_imgfiles[index]

        posefile = os.path.join(scan_dir, "poses.npy")

        rgb_intrinsic = np.loadtxt(
            os.path.join(scan_dir, "intrinsic/intrinsic_color.txt")
        )
        depth_intrinsic = np.loadtxt(
            os.path.join(scan_dir, "intrinsic/intrinsic_depth.txt")
        )

        npz_16 = np.load(os.path.join(scan_dir, "tsdf_0.16.npz"))
        tsdf_vol = npz_16["tsdf_vol"]
        voxel_size = npz_16["voxel_size"]
        tsdf_minbounds = npz_16["minbounds"]

        extents = np.load(os.path.join(scan_dir, "extents.npy")).astype(np.float32)

        poses = np.load(posefile)

        bad_img_inds = np.any(np.isnan(poses) | np.isinf(poses), axis=(1, 2))
        poses = poses[~bad_img_inds]
        rgb_imgfiles = np.array(rgb_imgfiles)[~bad_img_inds]
        depth_imgfiles = np.array(depth_imgfiles)[~bad_img_inds]

        txtfile = os.path.join(scan_dir, os.path.basename(scan_dir) + ".txt")
        with open(txtfile, "r") as f:
            lines = f.read().split("\n")
        axis_alignment = np.fromstring(
            lines[0].split("=")[1], dtype=float, sep=" "
        ).reshape(4, 4)
        poses = (axis_alignment @ poses).astype(np.float32)

        img_inds = np.round(
            np.linspace(0, len(rgb_imgfiles), 25, endpoint=False)
        ).astype(int)
        img_inds = (img_inds + np.random.randint(img_inds[1])) % len(rgb_imgfiles)

        depth_imgs = (
            np.stack(
                [imageio.imread(depth_imgfiles[i]) for i in img_inds], axis=0
            ).astype(np.float32)
            / 1000
        )
        rgb_imgs = np.stack(
            [
                cv2.imread(rgb_imgfiles[i], cv2.IMREAD_REDUCED_COLOR_2)[..., [2, 1, 0]]
                for i in img_inds
            ],
            axis=0,
        )

        imheight, imwidth = depth_imgs[0].shape[:2]
        rgb_imgs_resized = np.stack(
            [cv2.resize(img, (imwidth, imheight)) for img in rgb_imgs], axis=0
        )

        pts = []
        rgb = []

        uu, vv = np.meshgrid(
            np.arange(imwidth, dtype=np.float32), np.arange(imheight, dtype=np.float32)
        )
        uuvv = np.stack((uu, vv), axis=-1)

        inv_intr = np.linalg.inv(depth_intrinsic[:3, :3]).astype(np.float32)
        for i in range(len(rgb_imgs)):
            inds = depth_imgs[i] > 0
            u = uu[inds]
            v = vv[inds]
            ranges = depth_imgs[i][inds]

            pose = poses[img_inds[i]]

            colors = np.stack(
                (
                    rgb_imgs_resized[i, :, :, 0][inds],
                    rgb_imgs_resized[i, :, :, 1][inds],
                    rgb_imgs_resized[i, :, :, 2][inds],
                ),
                axis=-1,
            )

            ones = np.ones(len(u), dtype=np.float32)
            xyz_cam = ranges[:, None] * (inv_intr @ np.c_[u, v, ones].T).T
            xyz_world = (pose @ np.c_[xyz_cam, ones].T).T[:, :3]

            pts.append(xyz_world)
            rgb.append(colors)

        pts = np.concatenate(pts, axis=0)
        rgb = np.concatenate(rgb, axis=0)

        inds = (pts < extents[1]) & (pts > extents[0])
        inds = inds[:, 0] & inds[:, 1] & inds[:, 2]

        pts = pts[inds]
        rgb = rgb[inds]

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64)))
        pcd.colors = o3d.utility.Vector3dVector(rgb / 255)
        sparse_pcd = pcd.voxel_down_sample(0.1)
        pts = np.asarray(sparse_pcd.points).astype(np.float32)
        rgb = np.asarray(sparse_pcd.colors).astype(np.float32)

        inds = np.meshgrid(
            np.arange(tsdf_vol.shape[1]),
            np.arange(tsdf_vol.shape[0]),
            np.arange(tsdf_vol.shape[2]),
        )
        inds = np.stack(
            (inds[1].flatten(), inds[0].flatten(), inds[2].flatten()), axis=1
        )
        query_coords = inds * voxel_size + tsdf_minbounds
        query_tsdf = tsdf_vol.flatten()

        # extents of the ScanNet reconstructed groundtruth -- tighter boundaries than
        # the homegrown TSDF reconstruction
        inds = np.all((query_coords < extents[1]) & (query_coords > extents[0]), axis=1)
        query_coords = query_coords[inds]
        query_tsdf = query_tsdf[inds]

        inds = query_tsdf < 1
        query_coords = query_coords[inds]
        query_tsdf = query_tsdf[inds]

        m = np.mean(pts, axis=0)
        pts -= m
        query_coords -= m
        pts /= [5, 5, .5]
        query_coords /= [5, 5, .5]

        if self.split == "train":
            rotmat = scipy.spatial.transform.Rotation.from_rotvec(
                np.array([0, 0, 1]) * np.pi * np.random.randint(4)
            ).as_matrix()
            flipmat = np.array(
                [
                    [np.random.choice((-1, 1)), 0, 0],
                    [0, np.random.choice((-1, 1)), 0],
                    [0, 0, 1],
                ]
            )
            pts = (rotmat @ flipmat @ pts.T).T
            query_coords = (rotmat @ flipmat @ query_coords.T).T

            maxpts = 2 ** 14
            if len(pts) > maxpts:
                inds = np.arange(len(pts), dtype=int)
                inds = np.random.choice(inds, replace=False, size=maxpts)
                pts = pts[inds]
                rgb = rgb[inds]
            else:
                n = maxpts - len(pts)
                pts = np.concatenate((pts, -100 * np.ones((n, 3))), axis=0)
                rgb = np.concatenate((rgb, -100 * np.ones((n, 3))), axis=0)

            maxqueries = 2 ** 13
            if len(query_coords) > maxqueries:
                inds = np.arange(len(query_coords), dtype=int)
                inds = np.random.choice(inds, replace=False, size=maxqueries)
                query_coords = query_coords[inds]
                query_tsdf = query_tsdf[inds]
            else:
                n = maxqueries - len(query_coords)
                query_coords = np.concatenate((query_coords, -100 * np.ones((n, 3))), axis=0)
                query_tsdf = np.concatenate((query_tsdf, 100 * np.ones(n)), axis=0)

        batch = [
            pts.astype(np.float32),
            rgb.astype(np.float32),
            query_coords.astype(np.float32),
            query_tsdf,
        ]

        if self.split == "test":
            gt_mesh_verts = npz_16["gt_mesh_verts"]
            gt_mesh_faces = npz_16["gt_mesh_faces"]
            gt_mesh_vertex_colors = npz_16["gt_mesh_vertex_colors"]
            # gt_mesh_verts += 10
            batch += [
                gt_mesh_verts,
                gt_mesh_faces,
                gt_mesh_vertex_colors,
            ]

        return batch

        """
        
        
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(gt_mesh_verts),
            o3d.utility.Vector3iVector(gt_mesh_faces),
        )
        mesh.vertex_colors = o3d.utility.Vector3dVector(gt_mesh_vertex_colors / 255)
        
        inds = np.argwhere(tsdf_vol < 1)
        tsdf = tsdf_vol[inds[:, 0], inds[:, 1], inds[:, 2]]
        colors = plt.cm.jet(tsdf * 0.5 + 0.5)[:, :3]
        coords = inds * voxel_size + minbounds
        tsdf_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(coords))
        tsdf_pcd.colors = o3d.utility.Vector3dVector(colors)
        
        geoms = [mesh, pcd, tsdf_pcd, pcd2]
        visibility = [True] * len(geoms)
        
        def toggle_geom(vis, geom_ind):
            if visibility[geom_ind]:
                vis.remove_geometry(geoms[geom_ind], reset_bounding_box=False)
                visibility[geom_ind] = False
            else:
                vis.add_geometry(geoms[geom_ind], reset_bounding_box=False)
                visibility[geom_ind] = True
        
        
        callbacks = {}
        for i in range(len(geoms)):
            callbacks[ord(str(i + 1))] = functools.partial(toggle_geom, geom_ind=i)
        o3d.visualization.draw_geometries_with_key_callbacks(geoms, callbacks)
        """


if __name__ == "__main__":
    scannet_dir = "/home/noah/data/scannet"
    scan_dirs = sorted(glob.glob(os.path.join(scannet_dir, "*")))

    dset = Dataset(scan_dirs, 10, split="train")
    self = dset
    index = 0
    a = []
    b = []
    for i in range(10):
        pts, rgb, query_coords, query_tsdf = dset[i]
        a.append(np.mean(pts[pts[:, 0] > -50], axis=0))
        b.append(np.var(pts[pts[:, 0] > -50], axis=0))


    query_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(query_coords))
    query_pcd.colors = o3d.utility.Vector3dVector(plt.cm.jet(query_tsdf)[:, :3])

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    geoms = [pcd, query_pcd]
    visibility = [True] * len(geoms)

    def toggle_geom(vis, geom_ind):
        if visibility[geom_ind]:
            vis.remove_geometry(geoms[geom_ind], reset_bounding_box=False)
            visibility[geom_ind] = False
        else:
            vis.add_geometry(geoms[geom_ind], reset_bounding_box=False)
            visibility[geom_ind] = True

    callbacks = {}
    for i in range(len(geoms)):
        callbacks[ord(str(i + 1))] = functools.partial(toggle_geom, geom_ind=i)
    o3d.visualization.draw_geometries_with_key_callbacks(geoms, callbacks)
