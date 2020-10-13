import glob
import os

import cv2
import numpy as np
import PIL.Image
import scipy.spatial
import torch
import trimesh

import fpn


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.dsetdir = "dset"

        self.rgb_imgfiles = sorted(glob.glob(os.path.join(self.dsetdir, "rgb/*.jpg")))
        self.depth_imgfiles = sorted(
            glob.glob(os.path.join(self.dsetdir, "depth/*.png"))
        )
        self.camera_poses = np.load(os.path.join(self.dsetdir, "camera_poses.npy"))
        self.intrinsic = np.load(os.path.join(self.dsetdir, "intrinsic.npy"))
        self.inv_intrinsic = np.linalg.inv(self.intrinsic)

        query_npz = np.load(os.path.join(self.dsetdir, "sdf.npz"))
        self.query_pts = query_npz["pts"]
        self.query_sdf = query_npz["sd"]

        self.mesh = trimesh.load("fuze.obj")
        v = np.asarray(self.mesh.vertices)
        vertmean = np.mean(v, axis=0)
        diaglen = np.linalg.norm(np.max(v, axis=0) - np.min(v, axis=0))
        v = (v - vertmean) / diaglen
        self.mesh.vertices = v

        self.n_pts = 256
        self.query_radius = 0.2

        self.n_anchors = 8

    def __len__(self):
        return len(self.rgb_imgfiles)

    def __getitem__(self, index):
        depth_img = (
            cv2.imread(self.depth_imgfiles[index], cv2.IMREAD_ANYDEPTH).astype(
                np.float32
            )
            / 1000
        )
        pil_img = PIL.Image.open(self.rgb_imgfiles[index])
        rgb_img = np.asarray(pil_img).copy()
        rgb_img_t = fpn.transform(pil_img)

        camera_pose = self.camera_poses[index]

        uv = np.argwhere(depth_img > 0)[:, [1, 0]]
        inds = np.arange(len(uv))
        anchor_inds = uv[np.random.choice(inds, size=self.n_anchors, replace=False)]

        anchor_uv = anchor_inds
        anchor_range = depth_img[anchor_inds[:, 1], anchor_inds[:, 0]]
        anchor_xyz_cam = (
            (self.inv_intrinsic @ np.c_[anchor_uv, np.ones(len(anchor_uv))].T)
            * -anchor_range
        ).T

        anchor_xyz = (
            camera_pose @ np.c_[anchor_xyz_cam, np.ones(len(anchor_xyz_cam))].T
        ).T[:, :3]

        in_range_inds = (
            np.linalg.norm(anchor_xyz[:, None] - self.query_pts, axis=-1)
            < self.query_radius
        )

        query_tsdf = []
        query_xyz_cam = []
        query_inds = []
        for i in range(self.n_anchors):
            inds = np.random.choice(
                np.argwhere(in_range_inds[i]).flatten(), replace=False, size=self.n_pts,
            )
            query_inds.append(inds)
            query_pts = self.query_pts[inds]
            query_tsdf.append(self.query_sdf[inds])
            query_xyz_cam.append(
                (np.linalg.inv(camera_pose) @ np.c_[query_pts, np.ones(len(query_pts))].T).T[:, :3]
            )

        query_inds = np.stack(query_inds, axis=0)
        query_xyz_cam = np.stack(query_xyz_cam, axis=0)
        query_sd = np.stack(query_tsdf, axis=0)
        query_occ = query_sd > 0

        anchor_cam_unit = anchor_xyz_cam / np.linalg.norm(
            anchor_xyz_cam, axis=-1, keepdims=True
        )
        center_pixel_u = np.array([0, 0, -1])

        xz_plane_projection = np.array([1, 0, 1]) * anchor_cam_unit
        xz_plane_projection /= np.linalg.norm(
            xz_plane_projection, axis=-1, keepdims=True
        )
        rot_axis = np.cross(center_pixel_u, xz_plane_projection)
        rot_axis /= np.linalg.norm(rot_axis, axis=-1, keepdims=True)
        horiz_angle = np.arccos(np.dot(xz_plane_projection, center_pixel_u))
        horiz_rot = scipy.spatial.transform.Rotation.from_rotvec(
            rot_axis * horiz_angle[:, None]
        ).as_matrix()

        yz_plane_projection = np.array([0, 1, 1]) * (
            (np.linalg.inv(horiz_rot) @ anchor_cam_unit.T)[
                np.arange(self.n_anchors), :, np.arange(self.n_anchors)
            ]
        )
        yz_plane_projection /= np.linalg.norm(
            yz_plane_projection, axis=-1, keepdims=True
        )
        rot_axis = np.cross(center_pixel_u, yz_plane_projection)
        rot_axis /= np.linalg.norm(rot_axis, axis=-1, keepdims=True)
        vert_angle = np.arccos(np.dot(yz_plane_projection, center_pixel_u))
        vert_rot = scipy.spatial.transform.Rotation.from_rotvec(
            rot_axis * vert_angle[:, None]
        ).as_matrix()

        cam2anchor_rot = horiz_rot @ vert_rot

        """
        todo
        """

        """
        cross = np.cross(center_pixel_u, anchor_cam_unit)
        cross /= np.linalg.norm(cross, axis=-1, keepdims=True)
        dot = np.dot(center_pixel_u, anchor_cam_unit.T)
        axis = cross
        if np.any(np.isnan(axis)):
            return self[np.random.randint(0, len(self))]
        angle = np.arccos(dot) 
        cam2anchor_rot = scipy.spatial.transform.Rotation.from_rotvec(axis * angle[:, None]).as_matrix()
        """

        query_xyz_cam_rotated = np.stack(
            [
                (np.linalg.inv(cam2anchor_rot[i]) @ query_xyz_cam[i].T).T
                for i in range(self.n_anchors)
            ],
            axis=0,
        )
        anchor_xyz_cam_rotated = (np.linalg.inv(cam2anchor_rot) @ anchor_xyz_cam.T)[
            np.arange(self.n_anchors), :, np.arange(self.n_anchors)
        ]

        query_coords = (
            query_xyz_cam_rotated - anchor_xyz_cam_rotated[:, None]
        ) / self.query_radius

        """

        verts_cam = (np.linalg.inv(camera_pose) @ np.c_[self.mesh.vertices, np.ones(len(self.mesh.vertices))].T).T[:, :3]
        uv = np.argwhere(depth_img > 0)[:, [1, 0]]
        ranges = depth_img[uv[:, 1], uv[:, 0]]
        xyz_cam = (self.inv_intrinsic @ np.c_[uv, np.ones(len(uv))].T).T * -ranges[:, None]
        t = camera_pose[:3, 3]

        gcf().add_subplot(111, projection='3d')
        plot(verts_cam[:, 0], verts_cam[:, 1], verts_cam[:, 2], '.')
        plot(xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2], '.')

        j = 0
        figure()
        subplot(221)
        imshow(rgb_img)
        plot(anchor_uv[j, 0], anchor_uv[j, 1], 'r.')
        gcf().add_subplot(222, projection='3d')
        plot([anchor_xyz_cam[j, 0]], [anchor_xyz_cam[j, 1]], [anchor_xyz_cam[j, 2]], '.')
        # plot([t[0]], [t[1]], [t[2]], '.')
        plot([0],[0], [0], '.')
        plot(verts_cam[:, 0], verts_cam[:, 1], verts_cam[:, 2], 'k.', markersize=.1)
        plot(xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2], '.')
        plot(query_xyz_cam[j, query_occ[j], 0], query_xyz_cam[j, query_occ[j], 1], query_xyz_cam[j, query_occ[j], 2], '.')
        plot(query_xyz_cam[j, ~query_occ[j], 0], query_xyz_cam[j, ~query_occ[j], 1], query_xyz_cam[j, ~query_occ[j], 2], '.')
        xlabel('x')
        ylabel('y')
        gcf().add_subplot(223, projection='3d')
        plot(query_xyz_cam_rotated[j, query_occ[j], 0], query_xyz_cam_rotated[j, query_occ[j], 1], query_xyz_cam_rotated[j, query_occ[j], 2], '.')
        plot(query_xyz_cam_rotated[j, ~query_occ[j], 0], query_xyz_cam_rotated[j, ~query_occ[j], 1], query_xyz_cam_rotated[j, ~query_occ[j], 2], '.')
        plot([anchor_xyz_cam_rotated[j, 0]], [anchor_xyz_cam_rotated[j, 1]], [anchor_xyz_cam_rotated[j, 2]], '.')
        plot([0], [0], [0], '.')
        xlabel('x')
        ylabel('y')
        gcf().add_subplot(224, projection='3d')
        plot(query_coords[j, query_occ[j], 0], query_coords[j, query_occ[j], 1], query_coords[j, query_occ[j], 2], '.')
        plot(query_coords[j, ~query_occ[j], 0], query_coords[j, ~query_occ[j], 1], query_coords[j, ~query_occ[j], 2], '.')
        plot([0], [0], [0], '.')
        xlabel('x')
        ylabel('y')
        """

        return (
            rgb_img,
            rgb_img_t,
            depth_img,
            anchor_uv,
            anchor_xyz_cam,
            anchor_xyz_cam_rotated,
            query_xyz_cam,
            query_xyz_cam_rotated,
            query_coords.astype(np.float32),
            query_occ,
            query_sd,
            query_inds,
            camera_pose,
            cam2anchor_rot,
            index,
        )

if __name__ == "__main__":
    self = Dataset()
    index = 0
