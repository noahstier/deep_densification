import importlib
import os

import igl
import open3d as o3d
import PIL.Image
import pyrender
import scipy.spatial
import trimesh
import tqdm

import config

spec = importlib.util.spec_from_file_location(
    "colmap_reader", config.colmap_reader_script
)
colmap_reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(colmap_reader)

house_dir = "/home/noah/data/suncg_output/00b0f09923d425d20047ee7e861ffc71"

rgb_imgdir = os.path.join(house_dir, "imgs/color")

sfm_ptfile = os.path.join(house_dir, "sfm/sparse/auto/points3D.bin")
sfm_imfile = os.path.join(house_dir, "sfm/sparse/auto/images.bin")
delaunay_meshfile = os.path.join(house_dir, "sfm/sparse/auto/meshed-delaunay.ply")
smoothed_meshfile = os.path.join(house_dir, "sfm/sparse/auto/smoothed-delaunay.ply")

delaunay_mesh = o3d.io.read_triangle_mesh(delaunay_meshfile)
smoothed_mesh = delaunay_mesh.filter_smooth_laplacian()

o3d.io.write_triangle_mesh(smoothed_meshfile, smoothed_mesh)

house_name = os.path.basename(house_dir)
gt_meshfile = os.path.join("/home/noah/data/suncg/suncg/house", house_name, "house.obj")
gt_mesh = o3d.io.read_triangle_mesh(gt_meshfile)

pts = colmap_reader.read_points3d_binary(sfm_ptfile)
ims = colmap_reader.read_images_binary(sfm_imfile)

pts = {
    pt_id: pt for pt_id, pt in pts.items() if pt.error < 1 and len(pt.image_ids) >= 5
}
pt_ids = sorted(pts.keys())
im_ids = sorted(ims.keys())

sfm_xyz = np.stack([pts[i].xyz for i in pt_ids], axis=0)
sfm_rgb = np.stack([pts[i].rgb for i in pt_ids], axis=0) / 255

pil_imgs = [PIL.Image.open(os.path.join(rgb_imgdir, ims[i].name)) for i in im_ids]
imgs = np.stack([np.asarray(img) for img in pil_imgs], axis=0)
imheight, imwidth, _ = imgs[0].shape

fusion_npz = np.load(os.path.join(house_dir, "fusion.npz"))
gt_xyz = fusion_npz["tsdf_point_cloud"][:, :3]
gt_rgb = fusion_npz["tsdf_point_cloud"][:, 3:]
gt_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt_xyz))
gt_pcd.colors = o3d.utility.Vector3dVector(gt_rgb / 255)


intr_file = os.path.join(house_dir, "camera_intrinsics")
camera_intrinsic = np.loadtxt(intr_file)[0].reshape(3, 3)
camera_extrinsics = np.zeros((len(imgs), 4, 4))
for i, im_id in enumerate(im_ids):
    im = ims[im_id]
    qw, qx, qy, qz = im.qvec
    r = scipy.spatial.transform.Rotation.from_quat([qx, qy, qz, qw]).as_matrix().T
    t = np.linalg.inv(-r.T) @ im.tvec
    camera_extrinsics[i] = np.array(
        [[*r[0], t[0]], [*r[1], t[1]], [*r[2], t[2]], [0, 0, 0, 1]]
    )

img_ind = 39

cam_pos = camera_extrinsics[img_ind, :3, 3]

sfm_xyz_cam = (
    np.linalg.inv(camera_extrinsics[img_ind]) @ np.c_[sfm_xyz, np.ones(len(sfm_xyz))].T
).T[:, :3]
sfm_uv = (camera_intrinsic @ sfm_xyz_cam.T).T
sfm_uv = sfm_uv[:, :2] / sfm_uv[:, 2:]

in_frustum_inds = (
    (sfm_xyz_cam[:, 2] > 0)
    & (sfm_uv[:, 0] >= 0)
    & (sfm_uv[:, 0] <= imwidth)
    & (sfm_uv[:, 1] >= 0)
    & (sfm_uv[:, 1] <= imheight)
)


def render_depth_img(mesh, extrinsic, intrinsic):
    fuze_trimesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
    mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
    scene = pyrender.Scene()
    scene.add(mesh)
    camera = pyrender.IntrinsicsCamera(intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2])
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = extrinsic[:3, 3]
    camera_pose[:3, :3] = extrinsic[:3, :3]
    r = scipy.spatial.transform.Rotation.from_rotvec(np.array([1, 0, 0]) * np.pi).as_matrix()
    camera_pose[:3, :3] = camera_pose[:3, :3] @ r
    scene.add(camera, pose=camera_pose)
    light = pyrender.SpotLight(
        color=np.ones(3),
        intensity=100,
        innerConeAngle=np.pi / 16.0,
        outerConeAngle=np.pi / 6.0,
    )
    scene.add(light, pose=camera_pose)
    r = pyrender.OffscreenRenderer(640, 480)
    _, depth = r.render(scene)
    return depth

est_depth_img = render_depth_img(smoothed_mesh, camera_extrinsics[img_ind], camera_intrinsic)
true_depth_img = render_depth_img(gt_mesh, camera_extrinsics[img_ind], camera_intrinsic)

sfm_depth = sfm_xyz_cam[in_frustum_inds, 2]
sfm_uv_inds = np.clip(np.round(sfm_uv[in_frustum_inds]).astype(int), [0, 0], [imwidth - 1, imheight - 1])
true_depth = true_depth_img[sfm_uv_inds[:, 1], sfm_uv_inds[:, 0]]
est_depth = est_depth_img[sfm_uv_inds[:, 1], sfm_uv_inds[:, 0]]

est_visible_inds = sfm_depth < est_depth + .01
real_visible_inds = sfm_depth < true_depth + .01

'''
verts = np.asarray(smoothed_mesh.vertices).astype(np.float64)
faces = np.asarray(smoothed_mesh.triangles).astype(np.int32)

dests = sfm_xyz[in_frustum_inds]
origins = np.tile(cam_pos[None], (len(dests), 1))
direcs = dests - origins
est_intersector = trimesh.ray.ray_triangle.RayMeshIntersector(
    trimesh.Trimesh(vertices=verts, faces=faces)
)
real_intersector = trimesh.ray.ray_triangle.RayMeshIntersector(
    trimesh.Trimesh(
        vertices=np.asarray(gt_mesh.vertices).astype(np.float64),
        faces=np.asarray(gt_mesh.triangles).astype(np.int32),
    )
)
est_visible_inds = np.zeros(len(dests), dtype=bool)
real_visible_inds = np.zeros(len(dests), dtype=bool)
for i in tqdm.trange(len(origins)):

    locations, ray_idx, tri_idx = est_intersector.intersects_location(
        origins[i : i + 1], direcs[i : i + 1]
    )
    intersect_dist = np.min(np.linalg.norm(locations - cam_pos, axis=1))
    pt_dist = np.linalg.norm(cam_pos - dests[i])
    if intersect_dist + 0.01 > pt_dist:
        est_visible_inds[i] = True

    locations, ray_idx, tri_idx = real_intersector.intersects_location(
        origins[i : i + 1], direcs[i : i + 1]
    )
    intersect_dist = np.min(np.linalg.norm(locations - cam_pos, axis=1))
    pt_dist = np.linalg.norm(cam_pos - dests[i])
    if intersect_dist + 0.01 > pt_dist:
        real_visible_inds[i] = True
'''

cam_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array([[*cam_pos]])))
cam_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
cam_mesh.translate(cam_pos, relative=False)
cam_mesh.paint_uniform_color(np.array([0, 0, 1], dtype=np.float64))
sfm_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sfm_xyz))
sfm_pcd.colors = o3d.utility.Vector3dVector(sfm_rgb)

in_frustum_pcd = o3d.geometry.PointCloud(
    o3d.utility.Vector3dVector(sfm_xyz[in_frustum_inds])
)
in_frustum_pcd.colors = o3d.utility.Vector3dVector(sfm_rgb[in_frustum_inds])
est_visible_pcd = o3d.geometry.PointCloud(
    o3d.utility.Vector3dVector(sfm_xyz[in_frustum_inds][est_visible_inds])
)
est_visible_pcd.colors = o3d.utility.Vector3dVector(
    sfm_rgb[in_frustum_inds][est_visible_inds]
)
real_visible_pcd = o3d.geometry.PointCloud(
    o3d.utility.Vector3dVector(sfm_xyz[in_frustum_inds][real_visible_inds])
)
real_visible_pcd.colors = o3d.utility.Vector3dVector(
    sfm_rgb[in_frustum_inds][real_visible_inds]
)
lines = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
    cam_pcd, est_visible_pcd, [(0, i) for i in range(len(est_visible_pcd.points))]
)
lines.paint_uniform_color(np.array([1, 0, 0], dtype=np.float64))

o3d.visualization.draw_geometries(
    [lines, cam_mesh, sfm_pcd, smoothed_mesh], mesh_show_back_face=True
)

matched_xyz = np.stack(
    [pts[pt_id].xyz for pt_id in ims[im_ids[img_ind]].point3D_ids if pt_id in pts],
    axis=0,
)
matched_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(matched_xyz))

matched_pcd.paint_uniform_color(np.array([1, 0, 0], dtype=np.float64))
est_visible_pcd.paint_uniform_color(np.array([0, 1, 0], dtype=np.float64))
real_visible_pcd.paint_uniform_color(np.array([1, 1, 0], dtype=np.float64))
sfm_pcd.paint_uniform_color(np.array([0, 0, 1], dtype=np.float64))

smoothed_mesh.compute_vertex_normals()


geoms = [
    smoothed_mesh,
    gt_pcd,
    cam_mesh,
    matched_pcd,
    est_visible_pcd,
    real_visible_pcd,
    sfm_pcd,
]
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
