import os

import cv2
import h5py
import numpy as np
import open3d as o3d
import scipy.spatial
import trimesh
import pyrender
import matplotlib.pyplot as plt
import tqdm

n = 10_000

fuze_trimesh = trimesh.load("fuze.obj")
verts = np.asarray(fuze_trimesh.vertices)
vertmean = np.mean(verts, axis=0)
diaglen = np.linalg.norm(np.max(verts, axis=0) - np.min(verts, axis=0))
normed_verts = (verts - vertmean) / diaglen
normed_verts_h = np.c_[normed_verts, np.ones(len(normed_verts))].T

fx = fy = 585.756
w = 224
h = 224
cx = w / 2
cy = h / 2

fovy = 2 * np.arctan(cy / fy)
fovx = 2 * np.arctan(cx / fx)

intrinsic = np.array([[-fx, 0, cx], [0, fy, cy], [0, 0, 1],])

uvecs = np.asarray(o3d.geometry.TriangleMesh.create_sphere(resolution=5).vertices)
u = np.array([0.3, 0.4, 0.5])
u /= np.linalg.norm(u)
axis = np.cross(uvecs, u)
axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
angle = np.arccos(np.dot(uvecs, u))
rotations = scipy.spatial.transform.Rotation.from_rotvec(
    axis * angle[:, None]
).as_matrix()

# plot3()
# plot(uvecs[:, 0], uvecs[:, 1], uvecs[:, 2], '.')

# f = h5py.File("fuze.hdf5", "w")
# rgb_dset = f.create_dataset("rgb_imgs", (n, h, w, 3), dtype=np.uint8)
# depth_dset = f.create_dataset("depth_imgs", (n, h, w), dtype=np.uint16)
# pose_dset = f.create_dataset("pose", (n, 4, 4), dtype=np.float32)
# intrinsic_dset = f.create_dataset("intrinsic", (3, 3), dtype=np.float32)

# intrinsic_dset[:] = intrinsic

camera_pose = np.eye(4)

right = camera_pose[:3, 0]
up = camera_pose[:3, 1]
backward = camera_pose[:3, 2]
forward = -backward

renderer = pyrender.OffscreenRenderer(w, h)
scene = pyrender.Scene(bg_color=[0.4, 0.6, 0.6])
camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy)
scene.add(camera, pose=camera_pose)
light = pyrender.SpotLight(
    color=np.ones(3),
    intensity=100.0,
    innerConeAngle=max(fovx, fovy),
    outerConeAngle=max(fovx, fovy),
)
scene.add(light, pose=camera_pose)

os.mkdir("dset")
os.mkdir("dset/rgb")
os.mkdir("dset/depth")
np.save("dset/intrinsic.npy", intrinsic)


poses = []
rotinds = []
for i in tqdm.trange(n):
    depth = np.random.uniform(1, 4)
    mesh_center_uv = np.array(
        [np.random.uniform(0.1 * w, w * 0.9), np.random.uniform(0.1 * h, h * 0.9),]
    )
    mesh_center_xyz = (np.linalg.inv(intrinsic) @ [*mesh_center_uv, 1]) * -depth

    # facingaway = i % 2 == 0

    # if facingaway:
    #     r1 = np.eye(4)
    #     r2 = np.eye(4)
    #     t = np.eye(4)

    #     r1[:3, :3] = scipy.spatial.transform.Rotation.from_rotvec(
    #         [0, np.pi, 0]
    #     ).as_matrix()

    #     x = np.random.normal(0, 1, size=5)
    #     axis = x[:3] / np.linalg.norm(x, axis=-1, keepdims=True)
    #     axis /= np.linalg.norm(axis)
    #     angle = np.random.uniform(0, 30) * np.pi / 180
    #     rotvec = axis * angle
    #     r2[:3, :3] = scipy.spatial.transform.Rotation.from_rotvec(rotvec).as_matrix()
    #     t[:3, 3] = mesh_center_xyz

    #     pose = t @ r2 @ r1
    # else:
    #     r1 = np.eye(4)
    #     t = np.eye(4)

    #     r1[:3, :3] = scipy.spatial.transform.Rotation.random().as_matrix()
    #     t[:3, 3] = mesh_center_xyz

    #     pose = t @ r1
    r1 = np.eye(4)
    r2 = np.eye(4)
    t = np.eye(4)

    rotind = i % len(rotations)
    rotinds.append(rotind)
    r2[:3, :3] = scipy.spatial.transform.Rotation.from_rotvec(
        np.array([0, 0, 1]) * np.random.uniform(0, 2 * np.pi)
    ).as_matrix()
    r1[:3, :3] = rotations[rotind]
    t[:3, 3] = mesh_center_xyz

    pose = t @ r1 @ r2

    fuze_trimesh.vertices = (pose @ normed_verts_h).T[:, :3]
    mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)

    meshnode = scene.add(mesh)
    rgb_img, depth_img = renderer.render(scene)
    scene.remove_node(meshnode)

    # rgb_dset[i] = rgb_img
    # depth_dset[i] = (depth_img * 1000).astype(np.uint16)
    # pose_dset[i] = pose

    cv2.imwrite(
        "dset/rgb/{}.jpg".format(str(i).zfill(5)),
        cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB),
    )
    cv2.imwrite(
        "dset/depth/{}.png".format(str(i).zfill(5)),
        (depth_img * 1000).astype(np.uint16),
    )
    poses.append(pose)

    """
    xyz = np.asarray(fuze_trimesh.vertices)
    xyz_cam = (camera_pose @ np.c_[xyz, np.ones(len(xyz))].T).T[:, :3]
    uv = (intrinsic @ xyz_cam.T).T
    uv = uv[:, :2] / uv[:, 2:]
    inds = (
        (uv[:, 0] >= 0) &
        (uv[:, 1] >= 0) &
        (uv[:, 0] <= w) &
        (uv[:, 1] <= h)
    )
    uv = np.floor(uv[inds]).astype(int)
    proj_depth = np.zeros_like(depth_img)
    proj_depth[uv[:, 1], uv[:, 0]] = np.dot(xyz_cam[inds], forward)

    gcf().add_subplot(221, projection='3d')
    plot([0, right[0]], [0, right[1]], [0, right[2]], 'r')
    plot([0, up[0]], [0, up[1]], [0, up[2]], 'g')
    plot([0, forward[0]], [0, forward[1]], [0, forward[2]], 'b')
    plot([0], [0], [0], 'r.')
    plot(fuze_trimesh.vertices[:, 0], fuze_trimesh.vertices[:, 1], fuze_trimesh.vertices[:, 2], '.')
    xlabel('x')
    ylabel('y')
    subplot(222)
    imshow(rgb_img)
    subplot(223)
    imshow(proj_depth)
    """

np.save("dset/poses.npy", poses)
np.save("dset/rotinds.npy", rotinds)

fuze_trimesh.vertices = normed_verts
query_radius = 0.2
maxnorm = np.max(np.linalg.norm(fuze_trimesh.vertices, axis=-1))
d = maxnorm + query_radius
x = np.random.normal(0, 1, size=(200_000, 5))
query_pts = x[:, :3] / np.linalg.norm(x, axis=-1, keepdims=True) * d
dist = fuze_trimesh.nearest.signed_distance(query_pts)
np.savez("dset/sdf.npz", pts=query_pts, sd=dist)


query_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(query_pts))
query_pcd.colors = o3d.utility.Vector3dVector(
    plt.cm.jet((dist - np.min(dist)) / (np.max(dist) - np.min(dist)))[:, :3]
)
mesh = o3d.io.read_triangle_mesh("fuze.obj")
v = np.asarray(mesh.vertices)
v = (v - vertmean) / diaglen
mesh.vertices = o3d.utility.Vector3dVector(v)
o3d.visualization.draw_geometries([query_pcd, mesh], mesh_show_back_face=True)
