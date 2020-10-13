import os
import glob

import cv2
import imageio
import numpy as np
import open3d as o3d
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, scan_dirs, n_imgs):
        super().__init__()
        self.scan_dirs = scan_dirs
        self.n_imgs = n_imgs

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
        seen_vol = npz_16["seen_vol"]
        voxel_size = npz_16["voxel_size"]
        minbounds = npz_16["minbounds"]

        poses = np.load(posefile)

        txtfile = os.path.join(scan_dir, os.path.basename(scan_dir) + ".txt")
        with open(txtfile, "r") as f:
            lines = f.read().split("\n")
        axis_alignment = np.fromstring(
            lines[0].split("=")[1], dtype=float, sep=" "
        ).reshape(4, 4)
        poses = axis_alignment @ poses

        img_inds = np.round(np.linspace(0, len(rgb_imgfiles), 25, endpoint=False)).astype(int)
        img_inds = (img_inds + np.random.randint(img_inds[1])) % len(rgb_imgfiles)

        depth_imgs = (
            np.stack(
                [imageio.imread(depth_imgfiles[i]) for i in img_inds], axis=0
            ).astype(np.float32)
            / 1000
        )
        rgb_imgs = np.stack([imageio.imread(rgb_imgfiles[i]) for i in img_inds], axis=0)

        imheight, imwidth = depth_imgs[0].shape[:2]
        rgb_imgs_resized = np.stack(
            [cv2.resize(img, (imwidth, imheight)) for img in rgb_imgs], axis=0
        )

        pts = []
        rgb = []

        uu, vv = np.meshgrid(
            np.arange(imwidth, dtype=np.int32), np.arange(imheight, dtype=np.int32)
        )
        uuvv = np.stack((uu, vv), axis=-1)

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

            inv_intr = np.linalg.inv(depth_intrinsic[:3, :3])

            xyz_cam = ranges[:, None] * (inv_intr @ np.c_[u, v, np.ones(len(u))].T).T
            xyz_world = (
                 pose @ np.c_[xyz_cam, np.ones(len(xyz_cam))].T
            ).T[:, :3]

            pts.append(xyz_world)
            rgb.append(colors)

        pts = np.concatenate(pts, axis=0)
        rgb = np.concatenate(rgb, axis=0)

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        pcd.colors = o3d.utility.Vector3dVector(rgb / 255)
        pcd2 = pcd.voxel_down_sample(.1)
        pts = np.asarray(pcd2.points)
        rgb = np.asarray(pcd2.colors)

        inds = np.meshgrid(
            np.arange(tsdf_vol.shape[1]),
            np.arange(tsdf_vol.shape[0]),
            np.arange(tsdf_vol.shape[2]),
        )
        inds = np.stack(
            (inds[1].flatten(), inds[0].flatten(), inds[2].flatten()), axis=1
        )
        query_coords = inds * voxel_size + minbounds
        query_tsdf = tsdf_vol.flatten()

        # m = np.mean(poses[:, :3, 3], axis=0)
        m = np.minimum(np.min(pts, axis=0), np.min(query_coords, axis=0))
        pts -= m
        query_coords -= m

        # inds = np.arange(len(pts))
        # inds = np.random.choice(inds, size=2048, replace=False)
        # pts = pts[inds]
        # rgb = rgb[inds]

        # inds = np.arange(len(pts))
        # inds = np.random.choice(inds, size=512, replace=False)
        # query_coords = query_coords[inds]
        # query_tsdf = query_tsdf[inds]


        return pts.astype(np.float32), rgb.astype(np.float32), query_coords.astype(np.float32), query_tsdf

        """
        
        npz_02 = np.load(os.path.join(scan_dir, 'tsdf_0.02.npz'))
        gt_mesh_verts = npz_02["gt_mesh_verts"]
        gt_mesh_faces = npz_02["gt_mesh_faces"]
        gt_mesh_vertex_colors = npz_02["gt_mesh_vertex_colors"]
        
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

    dset = Dataset(scan_dirs, 10)
    pts, rgb, query_coords, query_tsdf = dset[0]

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
