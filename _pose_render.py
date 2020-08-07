import h5py
import numpy as np
import scipy.spatial
import trimesh
import pyrender
import matplotlib.pyplot as plt
import tqdm

n = 50_000

f = h5py.File("fuze.hdf5", "w")
rgb_dset = f.create_dataset("rgb_imgs", (n, 400, 400, 3), dtype=np.uint8)
depth_dset = f.create_dataset("depth_imgs", (n, 400, 400), dtype=np.float32)
pose_dset = f.create_dataset("pose", (n, 4, 4), dtype=np.float32)

fuze_trimesh = trimesh.load("fuze.obj")
verts = np.asarray(fuze_trimesh.vertices)
verts -= np.mean(verts, axis=0)
verts *= 1.5
fuze_trimesh.vertices = verts

rotmats = []

r = pyrender.OffscreenRenderer(400, 400)

for i in tqdm.trange(n):
    rotmat = scipy.spatial.transform.Rotation.random().as_matrix()
    rotmats.append(rotmat)
    t = np.array([0, 0, -.35])
    pose = np.concatenate((np.c_[rotmat, t], [[0, 0, 0, 1]]), axis=0)
    fuze_trimesh.apply_transform(pose)
    mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
    fuze_trimesh.apply_transform(np.linalg.inv(pose))
    scene = pyrender.Scene()
    scene.add(mesh)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    s = np.sqrt(2) / 2
    camera_pose = np.eye(4)
    scene.add(camera, pose=camera_pose)
    light = pyrender.SpotLight(
        color=np.ones(3),
        intensity=3.0,
        innerConeAngle=np.pi / 16.0,
        outerConeAngle=np.pi / 6.0,
    )
    scene.add(light, pose=camera_pose)
    color, depth = r.render(scene)

    rgb_dset[i] = color
    depth_dset[i] = depth
    pose_dset[i] = pose
