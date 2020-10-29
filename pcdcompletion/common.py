import os
import glob

import numpy as np
import skimage.measure
import torch

import pyrender
import trimesh


def predict_mesh(model, pts, rgb):
    minbounds = torch.min(pts, dim=0)[0].cpu().numpy()
    maxbounds = torch.max(pts, dim=0)[0].cpu().numpy()
    minbounds -= 0.05
    maxbounds += 0.05

    res = 0.25
    x = np.arange(minbounds[0], maxbounds[0], res)
    y = np.arange(minbounds[1], maxbounds[1], res)
    z = np.arange(minbounds[2], maxbounds[2], res)
    xx, yy, zz = np.meshgrid(x, y, z)
    query_coords = np.c_[xx.flatten(), yy.flatten(), zz.flatten()]
    query_vol = np.zeros(xx.shape)
    query_inds = np.argwhere(query_vol == 0)
    query_coords = torch.Tensor(query_coords[None]).to(pts.device)
    grid_center = np.array([np.mean(y), np.mean(x), np.mean(z),])

    logits = model(pts[None], rgb[None], query_coords)
    preds = torch.sigmoid(logits).detach().cpu().numpy()

    query_vol[query_inds[:, 0], query_inds[:, 1], query_inds[:, 2]] = preds

    verts, faces, normals, values = skimage.measure.marching_cubes(query_vol, level=0.5)
    verts = (verts - np.mean(query_inds, axis=0)) * res + grid_center
    verts = verts[:, [1, 0, 2]]
    return verts, faces, preds, query_coords


def render_mesh(verts, faces):
    mesh = pyrender.Mesh.from_trimesh(trimesh.Trimesh(vertices=verts, faces=faces))

    h = w = 600
    fx = fy = 400

    renderer = pyrender.OffscreenRenderer(w, h)
    scene = pyrender.Scene(bg_color=[0, 0, 100])
    camera = pyrender.IntrinsicsCamera(fx, fy, w / 2, h / 2)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=10.0)

    camera_pose = np.eye(4)
    camera_pose[:3, 3] = [0, 0, 5]

    scene.add(mesh)
    scene.add(camera, pose=camera_pose)
    scene.add(light, pose=camera_pose)
    rgb_img, depth_img = renderer.render(scene)
    return rgb_img


def positional_encoding(xyz, L):
    encoding = []
    for l in range(L):
        encoding.append(torch.sin(2 ** l * np.pi * xyz))
        encoding.append(torch.cos(2 ** l * np.pi * xyz))
    encoding = torch.cat(encoding, dim=-1)
    return encoding


def get_scan_dirs(scannet_dir, scene_types=None):
    scan_dirs = []
    for scan_dir in sorted(glob.glob(os.path.join(scannet_dir, "*"))):
        if not os.path.exists(os.path.join(scan_dir, "imgs.h5")):
            continue
        scan_name = os.path.basename(scan_dir)

        if scene_types is not None:
            txt = os.path.join(scan_dir, scan_name + ".txt")
            with open(txt, "r") as f:
                lines = f.read().split("\n")
            for line in lines:
                if "sceneType" in line:
                    break
            else:
                continue
            scene_type = line.split(" = ")[1]
            for ok_scene_type in scene_types:
                if scene_type == ok_scene_type:
                    break
            else:
                continue
        scan_dirs.append(scan_dir)
    return scan_dirs


adj_file = "adjectives.txt"
noun_file = "nouns.txt"


def get_run_name():
    with open(adj_file, "r") as f:
        adjectives = f.read().split()
    with open(noun_file, "r") as f:
        nouns = f.read().split()
    run_name = np.random.choice(adjectives) + "_" + np.random.choice(nouns)
    return run_name
