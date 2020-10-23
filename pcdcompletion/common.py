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

    res = 0.16
    x = np.arange(minbounds[0], maxbounds[0], res)
    y = np.arange(minbounds[1], maxbounds[1], res)
    z = np.arange(minbounds[2], maxbounds[2], res)
    xx, yy, zz = np.meshgrid(x, y, z)
    query_coords = np.c_[xx.flatten(), yy.flatten(), zz.flatten()]
    query_vol = np.zeros(xx.shape)
    query_inds = np.argwhere(query_vol == 0)
    query_coords = torch.Tensor(query_coords[None]).cuda()
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
    fx = fy = 500

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
