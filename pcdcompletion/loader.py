import os
import glob

import cv2
import h5py
import imageio
import numpy as np
import open3d as o3d
import scipy.spatial
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, scan_dirs, n_imgs=-1, augment=False, load_gt_mesh=False):
        super().__init__()
        self.scan_dirs = scan_dirs
        self.n_imgs = n_imgs
        self.augment = augment
        self.load_gt_mesh = load_gt_mesh


    def __len__(self):
        return len(self.scan_dirs)

    def __getitem__(self, index):
        index = 0
        scan_dir = self.scan_dirs[index]

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

        img_inds = np.arange(len(poses))
        if self.n_imgs != -1:
            img_inds = np.random.choice(img_inds, size=self.n_imgs, replace=False)

        img_dset = h5py.File(os.path.join(scan_dir, "imgs.h5"), "r")
        depth_img_dset = img_dset["depth_imgs"]
        rgb_img_dset = img_dset["rgb_imgs"]
        depth_imgs = (
            np.stack([depth_img_dset[i] for i in img_inds], axis=0).astype(np.float32)
            / 1000
        )
        rgb_imgs = np.stack([rgb_img_dset[i] for i in img_inds], axis=0)

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
        rgb = np.concatenate(rgb, axis=0) / 255

        minbounds = np.min(pts, axis=0)
        res = 0.1
        inds = np.floor((pts - minbounds) / res).astype(np.uint8)
        a = np.arange(len(pts), dtype=np.int32)
        vol = np.zeros(np.max(inds, axis=0) + 1, dtype=np.int32)
        vol[inds[:, 0], inds[:, 1], inds[:, 2]] = a
        inds = np.argwhere(vol > 0)
        inds = vol[inds[:, 0], inds[:, 1], inds[:, 2]]
        pts = pts[inds]
        rgb = rgb[inds]

        """
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64)))
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        sparse_pcd = pcd.voxel_down_sample(0.1)
        pts = np.asarray(sparse_pcd.points).astype(np.float32)
        rgb = np.asarray(sparse_pcd.colors).astype(np.float32)
        """

        inds = (pts < extents[1]) & (pts > extents[0])
        inds = inds[:, 0] & inds[:, 1] & inds[:, 2]

        if not np.all(inds):
            pts = pts[inds]
            rgb = rgb[inds]

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

        pcd_center = np.mean(pts, axis=0)
        pts -= pcd_center
        query_coords -= pcd_center

        if self.augment:
            angle = np.random.uniform(0, 2 * np.pi)  # np.pi * np.random.randint(4)
            rotmat = scipy.spatial.transform.Rotation.from_rotvec(
                np.array([0, 0, 1]) * angle
            ).as_matrix()
            flipmat = np.eye(3)
            flipmat[0, 0] = np.random.choice((-1, 1))
            pts = (rotmat @ flipmat @ pts.T).T
            query_coords = (rotmat @ flipmat @ query_coords.T).T

            pts += np.random.uniform(-0.005, 0.005, size=pts.shape)

            maxqueries = 2 ** 13
            if len(query_coords) > maxqueries:
                inds = np.arange(len(query_coords), dtype=int)
                inds = np.random.choice(inds, replace=False, size=maxqueries)
                query_coords = query_coords[inds]
                query_tsdf = query_tsdf[inds]
            else:
                n = maxqueries - len(query_coords)
                query_coords = np.concatenate(
                    (query_coords, -100 * np.ones((n, 3))), axis=0
                )
                query_tsdf = np.concatenate((query_tsdf, 100 * np.ones(n)), axis=0)

            maxpts = 2 ** 13
            if len(pts) > maxpts:
                inds = np.arange(len(pts), dtype=int)
                inds = np.random.choice(inds, replace=False, size=maxpts)
                pts = pts[inds]
                rgb = rgb[inds]
            else:
                n = maxpts - len(pts)
                pts = np.concatenate((pts, -100 * np.ones((n, 3))), axis=0)
                rgb = np.concatenate((rgb, -100 * np.ones((n, 3))), axis=0)

        batch = [
            pts.astype(np.float32),
            rgb.astype(np.float32),
            query_coords.astype(np.float32),
            query_tsdf,
        ]

        if self.load_gt_mesh:
            gt_mesh_verts = npz_16["gt_mesh_verts"]
            gt_mesh_faces = npz_16["gt_mesh_faces"]
            gt_mesh_vertex_colors = npz_16["gt_mesh_vertex_colors"]
            gt_mesh_verts -= pcd_center
            if self.augment:
                gt_mesh_verts = (rotmat @ flipmat @ gt_mesh_verts.T).T
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
    import config
    import tqdm

    scan_dirs = sorted(glob.glob(os.path.join(config.scannet_dir, "*")))

    dset = Dataset(scan_dirs[3:], n_imgs=17, augment=False, load_gt_mesh=True)
    self = dset
    index = 0
    (
        pts,
        rgb,
        query_coords,
        query_tsdf,
        gt_mesh_verts,
        gt_mesh_faces,
        gt_mesh_vertex_colors,
    ) = dset[index]

    gt_mesh_faces = np.concatenate((gt_mesh_faces, gt_mesh_faces[:, [2, 1, 0]]), axis=0)
    gt_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(gt_mesh_verts),
        o3d.utility.Vector3iVector(gt_mesh_faces),
    )
    gt_mesh.vertex_colors = o3d.utility.Vector3dVector(gt_mesh_vertex_colors / 255)
    gt_mesh.compute_vertex_normals()


    inds = pts[:, 0] > -50
    pts = pts[inds]
    rgb = rgb[inds]

    inds = query_tsdf < 50
    query_coords = query_coords[inds]
    query_tsdf = query_tsdf[inds]

    tsdf_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(query_coords))
    tsdf_pcd.colors = o3d.utility.Vector3dVector(plt.cm.jet(query_tsdf * .5 + .5)[:, :3])

    occ_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(query_coords))
    occ_pcd.colors = o3d.utility.Vector3dVector(plt.cm.jet((query_tsdf < 0).astype(np.float32))[:, :3])

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    geoms = [gt_mesh, pcd, tsdf_pcd, occ_pcd]
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
