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

n = 1_000

# meshfile = "/home/noah/data/shapenet/02691156/10e4331c34d610dacc14f1e6f4f4f49b/models/model_normalized.obj"
meshfile = "fuze.obj"

# fuze_trimesh = trimesh.load(meshfile)
# geoms = list(fuze_trimesh.geometry.values())
# fuze_trimesh= sum(geoms)


fuze_trimesh = trimesh.load(meshfile)
verts = np.asarray(fuze_trimesh.vertices)
vertmean = np.mean(verts, axis=0)
diaglen = np.linalg.norm(np.max(verts, axis=0) - np.min(verts, axis=0))
normed_verts = (verts - vertmean) / diaglen
fuze_trimesh.vertices = normed_verts

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
discrete_rotations = scipy.spatial.transform.Rotation.from_rotvec(
    axis * angle[:, None]
).as_matrix()

# gcf().add_subplot(111, projection='3d')
# plot(uvecs[:, 0], uvecs[:, 1], uvecs[:, 2], '.')

camera_pose = np.eye(4)

right = camera_pose[:3, 0]
up = camera_pose[:3, 1]
backward = camera_pose[:3, 2]
forward = -backward

renderer = pyrender.OffscreenRenderer(w, h)
scene = pyrender.Scene(bg_color=[0.4, 0.6, 0.6])
camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy)

mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
scene.add(mesh)

light = pyrender.DirectionalLight(color=np.ones(3), intensity=10.0)

# light = pyrender.SpotLight(
#     color=np.ones(3),
#     intensity=100.0,
#     innerConeAngle=max(fovx, fovy),
#     outerConeAngle=max(fovx, fovy),
# )

os.mkdir("dset")
os.mkdir("dset/rgb")
os.mkdir("dset/depth")
np.save("dset/intrinsic.npy", intrinsic)

use_discrete_poses = False
facingaway = False

camera_poses = []
discrete_pose_labels = []
for i in tqdm.trange(n):

    t_direction = np.random.normal(0, 1, size=3)
    t_direction /= np.linalg.norm(t_direction, keepdims=True)
    t_range = np.random.uniform(1, 4)
    cam_pos = t_direction * t_range

    t = np.eye(4)
    t[:3, 3] = cam_pos

    v = np.random.normal(0, 1, size=3)
    v /= np.linalg.norm(v, keepdims=True)

    backward = cam_pos / np.linalg.norm(cam_pos, keepdims=True)
    forward = -backward
    right = np.cross(v, forward)
    right /= np.linalg.norm(right, keepdims=True)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up, keepdims=True)

    r1 = np.eye(4)
    r1[:3, :3] = np.c_[right, up, backward]

    rvec = np.random.normal(0, 1, size=3)
    rvec = rvec / np.linalg.norm(rvec, keepdims=True) * np.random.uniform(0, np.pi / 16)
    r2 = np.eye(4)
    r2[:3, :3] = scipy.spatial.transform.Rotation.from_rotvec(rvec).as_matrix()

    camera_pose = t @ r1 @ r2

    """
    plot3()
    plot([0], [0], [0], '.')
    plot([t[0]], [t[1]], [t[2]], '.')
    plot(
        [t[0], t[0] + forward[0]], 
        [t[1], t[1] + forward[1]], 
        [t[2], t[2] + forward[2]], 
        'b'
    )
    plot(
        [t[0], t[0] + right[0]], 
        [t[1], t[1] + right[1]], 
        [t[2], t[2] + right[2]], 
        'r'
    )
    plot(
        [t[0], t[0] + up[0]], 
        [t[1], t[1] + up[1]], 
        [t[2], t[2] + up[2]], 
        'g'
    )
    xlabel('x')
    ylabel('y')
    """

    camera_node = scene.add(camera, pose=camera_pose)
    light_node = scene.add(light, pose=camera_pose)
    rgb_img, depth_img = renderer.render(scene)
    scene.remove_node(camera_node)
    scene.remove_node(light_node)

    cv2.imwrite(
        "dset/rgb/{}.jpg".format(str(i).zfill(5)),
        cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB),
    )
    cv2.imwrite(
        "dset/depth/{}.png".format(str(i).zfill(5)),
        (depth_img * 1000).astype(np.uint16),
    )
    camera_poses.append(camera_pose)

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

np.save("dset/camera_poses.npy", camera_poses)
if use_discrete_poses:
    np.save("dset/discrete_pose_labels.npy", discrete_pose_labels)

near_query_pts = fuze_trimesh.sample(195_000)
near_query_pts += np.random.normal(0, 0.01, size=near_query_pts.shape)

query_radius = 1
maxnorm = np.max(np.linalg.norm(fuze_trimesh.vertices, axis=-1))
d = maxnorm + query_radius
x = np.random.normal(0, 1, size=(5000, 5))
far_query_pts = x[:, :3] / np.linalg.norm(x, axis=-1, keepdims=True) * d

query_pts = np.concatenate((near_query_pts, far_query_pts), axis=0)

dist = fuze_trimesh.nearest.signed_distance(query_pts)
np.savez("dset/sdf.npz", pts=query_pts, sd=dist)


"""
query_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(query_pts))
query_pcd.colors = o3d.utility.Vector3dVector(
    plt.cm.jet((dist - np.min(dist)) / (np.max(dist) - np.min(dist)))[:, :3]
)
mesh = o3d.io.read_triangle_mesh(meshfile)
v = np.asarray(mesh.vertices)
v = (v - vertmean) / diaglen
mesh.vertices = o3d.utility.Vector3dVector(v)
o3d.visualization.draw_geometries([query_pcd, mesh], mesh_show_back_face=True)
"""
