import os
import glob

import cv2
import h5py
import imageio
import numpy as np
import open3d as o3d
import scipy.spatial
import torch

import common


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        scan_dirs,
        n_imgs=-1,
        augment=False,
        load_gt_mesh=False,
        maxqueries=-1,
        maxpts=-1,
    ):
        super().__init__()
        self.scan_dirs = scan_dirs
        self.n_imgs = n_imgs
        self.augment = augment
        self.load_gt_mesh = load_gt_mesh
        self.maxpts = maxpts
        self.maxqueries = maxqueries

    def __len__(self):
        return len(self.scan_dirs)

    def __getitem__(self, index):
        scan_dir = self.scan_dirs[index]

        posefile = os.path.join(scan_dir, "poses.npy")

        rgb_intrinsic = np.loadtxt(
            os.path.join(scan_dir, "intrinsic/intrinsic_color.txt")
        )
        depth_intrinsic = np.loadtxt(
            os.path.join(scan_dir, "intrinsic/intrinsic_depth.txt")
        )

        npz = np.load(os.path.join(scan_dir, "tsdf_0.16.npz"))
        tsdf_vol = npz["tsdf_vol"]
        seen_vol = npz["seen_vol"]
        voxel_size = npz["voxel_size"]
        tsdf_minbounds = npz["minbounds"]

        extents = np.load(os.path.join(scan_dir, "extents.npy")).astype(np.float32)

        poses = np.load(posefile)

        img_inds = np.arange(len(poses))
        if self.n_imgs != -1:
            img_inds = np.round(
                np.linspace(0, len(poses) - 1, self.n_imgs, endpoint=False)
            ).astype(int)
            # if self.augment:
            #     img_inds = (img_inds + np.random.randint(0, img_inds[1])) % len(poses)

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
        ).astype(np.float32) / 255

        pts = []
        rgb = []
        # pt_inds = []

        uu, vv = np.meshgrid(
            np.arange(imwidth, dtype=np.float32), np.arange(imheight, dtype=np.float32)
        )

        d = depth_imgs.reshape(len(depth_imgs), -1)
        r = rgb_imgs_resized.reshape(len(rgb_imgs_resized), -1, 3)
        uuu = uu.flatten()
        vvv = vv.flatten()

        inv_intr = np.linalg.inv(depth_intrinsic[:3, :3]).astype(np.float32)
        for i in range(len(depth_imgs)):

            inds = d[i] > 0
            # a = np.random.choice(np.argwhere(inds)[:, 0], size=min(1000, np.sum(inds)), replace=False)
            # inds = np.zeros(len(inds), dtype=bool)
            # inds[a] = True

            u = uuu[inds]
            v = vvv[inds]
            ranges = d[i][inds]

            pose = poses[img_inds[i]]
            colors = np.stack(
                (r[i, :, 0][inds], r[i, :, 1][inds], r[i, :, 2][inds],), axis=-1
            )

            ones = np.ones(len(u), dtype=np.float32)
            xyz_cam = ranges[:, None] * (inv_intr @ np.c_[u, v, ones].T).T
            xyz_world = (pose @ np.c_[xyz_cam, ones].T).T[:, :3]

            # pt_inds.append(np.stack((np.ones(len(u)) * i, u, v), axis=1))
            pts.append(xyz_world)
            rgb.append(colors)

        pts = np.concatenate(pts, axis=0)
        rgb = np.concatenate(rgb, axis=0)
        # pt_inds = np.concatenate(pt_inds, axis=0)

        minbounds = np.min(pts, axis=0)
        res = np.array([0.1], dtype=np.float32)
        inds = np.floor((pts - minbounds) / res).astype(np.int32)
        a = np.arange(len(pts), dtype=np.int32)
        vol = np.zeros(np.max(inds, axis=0) + 1, dtype=np.int32)
        vol[inds[:, 0], inds[:, 1], inds[:, 2]] = a
        inds = np.argwhere(vol > 0)
        inds = vol[inds[:, 0], inds[:, 1], inds[:, 2]]
        pts = pts[inds]
        rgb = rgb[inds]
        # pt_inds = pt_inds[inds]

        inds = (pts < extents[1]) & (pts > extents[0])
        inds = inds[:, 0] & inds[:, 1] & inds[:, 2]

        if not np.all(inds):
            pts = pts[inds]
            rgb = rgb[inds]
            # pt_inds = pt_inds[inds]

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

        inds = seen_vol.flatten()
        query_tsdf = query_tsdf[inds]
        query_coords = query_coords[inds]

        # extents of the ScanNet reconstructed groundtruth -- tighter boundaries than
        # the homegrown TSDF reconstruction
        inds = np.all((query_coords < extents[1]) & (query_coords > extents[0]), axis=1)
        query_coords = query_coords[inds]
        query_tsdf = query_tsdf[inds]

        # inds = query_tsdf < 1
        # query_coords = query_coords[inds]
        # query_tsdf = query_tsdf[inds]
        # inds = (np.abs(query_tsdf) < 1) & (np.abs(query_tsdf) > .1)
        inds = np.abs(query_tsdf) < 1
        query_coords = query_coords[inds]
        query_tsdf = query_tsdf[inds]

        if self.augment:
            # angle = np.random.uniform(0, 2 * np.pi)
            angle = np.pi / 2 * np.random.randint(4)
            rotmat = scipy.spatial.transform.Rotation.from_rotvec(
                np.array([0, 0, 1]) * angle
            ).as_matrix()
            flipmat = np.eye(3)
            flipmat[0, 0] = np.random.choice((-1, 1))
            pts = (rotmat @ flipmat @ pts.T).T
            query_coords = (rotmat @ flipmat @ query_coords.T).T

            # pts += np.random.uniform(-0.005, 0.005, size=pts.shape)
            # query_coords += np.random.uniform(-0.001, 0.001, size=query_coords.shape)

        if self.maxqueries > -1:
            if len(query_coords) > self.maxqueries:
                inds = np.arange(len(query_coords), dtype=int)
                inds = np.random.choice(inds, replace=False, size=self.maxqueries)
                query_coords = query_coords[inds]
                query_tsdf = query_tsdf[inds]
            else:
                n = self.maxqueries - len(query_coords)
                query_coords = np.concatenate(
                    (query_coords, -100 * np.ones((n, 3))), axis=0
                )
                query_tsdf = np.concatenate((query_tsdf, 100 * np.ones(n)), axis=0)

        if self.maxpts > -1:
            if len(pts) > self.maxpts:
                inds = np.arange(len(pts), dtype=int)
                inds = np.random.choice(inds, replace=False, size=self.maxpts)
                pts = pts[inds]
                rgb = rgb[inds]
                # pt_inds = pt_inds[inds]
                pcd_center = np.mean(pts, axis=0)
                pts -= pcd_center
                query_coords -= pcd_center
            else:
                pcd_center = np.mean(pts, axis=0)
                pts -= pcd_center
                query_coords -= pcd_center
                n = self.maxpts - len(pts)
                pts = np.concatenate((pts, -100 * np.ones((n, 3))), axis=0)
                rgb = np.concatenate((rgb, -100 * np.ones((n, 3))), axis=0)
                # pt_inds = np.concatenate((pt_inds, -100 * np.ones((n, 3))), axis=0)

        batch = [
            pts.astype(np.float32),
            rgb.astype(np.float32),
            query_coords.astype(np.float32),
            query_tsdf,
            # rgb_imgs_resized,
            # pt_inds,
        ]

        if self.load_gt_mesh:
            gt_mesh_verts = npz["gt_mesh_verts"]
            gt_mesh_faces = npz["gt_mesh_faces"]
            gt_mesh_vertex_colors = npz["gt_mesh_vertex_colors"]
            if self.augment:
                gt_mesh_verts = (rotmat @ flipmat @ gt_mesh_verts.T).T
            gt_mesh_verts -= pcd_center
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


def cache(dest_dir, dset, n):
    os.makedirs(dest_dir, exist_ok=True)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=1,
        num_workers=4,
        shuffle=False,
        worker_init_fn=lambda worker_id: np.random.seed(),
    )
    for i in tqdm.trange(n):
        for index, batch in enumerate(loader):
            # (pts, rgb, query_coords, query_tsdf, rgb_imgs, pt_inds) = batch
            (pts, rgb, query_coords, query_tsdf) = batch
            np.savez(
                os.path.join(
                    dest_dir, str(index).zfill(6) + "-" + str(i).zfill(6) + ".npz"
                ),
                pts=pts[0],
                rgb=rgb[0],
                query_coords=query_coords[0],
                query_tsdf=query_tsdf[0],
            )


class CachedDataset(torch.utils.data.Dataset):
    def __init__(self, dset_dir, augment=False):
        self.augment = augment
        npzfiles = sorted(glob.glob(os.path.join(dset_dir, "*.npz")))
        self.batch_dict = {}
        for npzfile in npzfiles:
            index = int(os.path.basename(npzfile).split("-")[0])
            n = int(os.path.basename(npzfile).split("-")[1].split(".")[0])
            if index not in self.batch_dict:
                self.batch_dict[index] = []
            self.batch_dict[index].append(npzfile)

    def __len__(self):
        return len(self.batch_dict)

    def __getitem__(self, index):
        if self.augment:
            npzfile = np.random.choice(self.batch_dict[index])
        else:
            npzfile = self.batch_dict[index][0]
        npz = np.load(npzfile)
        pts = npz["pts"]
        rgb = npz["rgb"]
        query_coords = npz["query_coords"]
        query_tsdf = npz["query_tsdf"]
        # rgb_imgs = npz["rgb_imgs"]
        # pt_inds = npz["pt_inds"]
        return pts, rgb, query_coords, query_tsdf # , rgb_imgs, pt_inds)


if __name__ == "__main__":
    import config
    import tqdm

    # dset = CachedDataset("batches", augment=True)
    # # plot3()
    # # plot(pts[:, 0], pts[:, 1], pts[:, 2], ".")

    scan_dirs = common.get_scan_dirs(
        config.scannet_dir,
        scene_types=[
            "Living room / Lounge",
            "Bathroom",
            "Bedroom / Hotel",
            "Kitchen",
            "Conference Room",
            "Copy/Mail Room",
        ],
    )
    scan_dirs = [d for d in scan_dirs if d.endswith("00")]

    # dset = Dataset(scan_dirs, n_imgs=-1, augment=False, load_gt_mesh=True, maxpts=-1, maxqueries=-1)
    dset = Dataset(
        scan_dirs,
        n_imgs=17,
        augment=True,
        load_gt_mesh=False,
        maxpts=config.maxpts,
        maxqueries=config.maxqueries,
    )
    self = dset
    index = 0
    (
        pts,
        rgb,
        query_coords,
        query_tsdf,
        rgb_imgs,
        pt_inds,
        gt_mesh_verts,
        gt_mesh_faces,
        gt_mesh_vertex_colors,
    ) = dset[index]

    gt_render = common.render_mesh(gt_mesh_verts, gt_mesh_faces)

    """
    plot3()
    plot(pts[:, 0], pts[:, 1], pts[:, 2], '.')
    """

    """
    m = []
    for index in tqdm.trange(len(dset)):
        batch = dset[index]
        pts = batch[0]
        _m = np.max(np.abs(pts[pts > -99]))
        if _m > 10:
            break
        m.append(_m)

    """

    gt_mesh_faces = np.concatenate((gt_mesh_faces, gt_mesh_faces[:, [2, 1, 0]]), axis=0)
    gt_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(gt_mesh_verts),
        o3d.utility.Vector3iVector(gt_mesh_faces),
    )
    gt_mesh.vertex_colors = o3d.utility.Vector3dVector(gt_mesh_vertex_colors)
    gt_mesh.compute_vertex_normals()

    inds = pts[:, 0] > -50
    pts = pts[inds]
    rgb = rgb[inds]

    inds = query_tsdf < 50
    query_coords = query_coords[inds]
    query_tsdf = query_tsdf[inds]

    tsdf_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(query_coords))
    tsdf_pcd.colors = o3d.utility.Vector3dVector(
        plt.cm.jet(query_tsdf * 0.5 + 0.5)[:, :3]
    )

    occ_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(query_coords))
    occ_pcd.colors = o3d.utility.Vector3dVector(
        plt.cm.jet((query_tsdf < 0).astype(np.float32))[:, :3]
    )

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    geoms = [gt_mesh, pcd, tsdf_pcd, occ_pcd]
    # geoms = [pcd, tsdf_pcd, occ_pcd]
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
