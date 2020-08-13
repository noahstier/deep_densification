import glob
import importlib
import os

import cv2
import h5py
import tqdm

colmap_reader_script = "/home/noah/Documents/colmap/scripts/python/read_write_model.py"
spec = importlib.util.spec_from_file_location("colmap_reader", colmap_reader_script)
colmap_reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(colmap_reader)

nmax = 10_000

dset = h5py.File("posed_patches.hdf5", "w")
patch_dset = dset.create_dataset(
    "patches", shape=(nmax, 2, 256, 256, 3), dtype=np.uint8
)
pt_dset = dset.create_dataset("pts", shape=(nmax, 3), dtype=np.float32)
cam_dset = dset.create_dataset("cams", shape=(nmax, 2, 3), dtype=np.float32)

house_dirs = sorted(glob.glob("/home/noah/data/suncg_output/*"))
np.random.shuffle(house_dirs)

n = 0

for i_house, house_dir in enumerate(tqdm.tqdm(house_dirs)):
    imfile = os.path.join(house_dir, "sfm/sparse/auto/images.bin")
    ptfile = os.path.join(house_dir, "sfm/sparse/auto/points3D.bin")

    try:
        ims = colmap_reader.read_images_binary(imfile)
        pts = colmap_reader.read_points3d_binary(ptfile)
    except Exception as e:
        print(e)
        continue

    imgs = {
        im_id: cv2.cvtColor(
            cv2.imread(os.path.join(house_dir, "imgs", "color", im.name)),
            cv2.COLOR_BGR2RGB,
        )
        for im_id, im in ims.items()
    }

    pts = {
        pt_id: pt
        for pt_id, pt in pts.items()
        if pt.error < 1 and len(pt.image_ids) >= 5
    }
    for pt_id, pt in pts.items():

        xy = np.stack(
            [
                ims[im_id].xys[pt_idx]
                for im_id, pt_idx in zip(pt.image_ids, pt.point2D_idxs)
            ],
            axis=0,
        )
        inds = np.all((xy > 128) & (xy < [640 - 128, 480 - 128]), axis=-1)
        inds = inds[None] & inds[:, None]

        cam_pos = np.stack(
            [-ims[im_id].qvec2rotmat().T @ ims[im_id].tvec for im_id in pt.image_ids],
            axis=0,
        )
        pt_to_cam = cam_pos - pt.xyz
        pt_to_cam_u = pt_to_cam / np.linalg.norm(pt_to_cam, axis=1, keepdims=True)
        angles = (
            np.arccos(
                np.clip(np.sum(pt_to_cam_u[None] * pt_to_cam_u[:, None], axis=-1), 0, 1)
            )
            * 180
            / np.pi
        )
        ranges = np.linalg.norm(pt_to_cam, axis=-1)
        range_diffs = np.abs(ranges[None] - ranges[:, None])

        angles *= 1 - np.tri(len(angles))
        range_diffs *= 1 - np.tri(len(range_diffs))
        inds &= ~np.tri(len(inds), dtype=bool)

        cam_pair_inds = np.argwhere(
            inds & (angles > 20) & (angles < 70) & (range_diffs < 0.2)
        )
        if len(cam_pair_inds) < 1:
            continue
        # cam_0, cam_1 = cam_pair_inds[np.random.randint(len(cam_pair_inds))]
        cam_0, cam_1 = cam_pair_inds[0]

        im_id_0 = pt.image_ids[cam_0]
        im_id_1 = pt.image_ids[cam_1]

        point2d_idx_0 = pt.point2D_idxs[cam_0]
        point2d_idx_1 = pt.point2D_idxs[cam_1]

        im_0 = ims[im_id_0]
        im_1 = ims[im_id_1]

        xy_0 = np.floor(im_0.xys[point2d_idx_0]).astype(int)
        xy_1 = np.floor(im_1.xys[point2d_idx_1]).astype(int)

        x_low_0, y_low_0 = xy_0 - 128
        x_high_0, y_high_0 = xy_0 + 128
        x_low_1, y_low_1 = xy_1 - 128
        x_high_1, y_high_1 = xy_1 + 128

        patch_0 = imgs[im_id_0][y_low_0:y_high_0, x_low_0:x_high_0]
        patch_1 = imgs[im_id_1][y_low_1:y_high_1, x_low_1:x_high_1]

        patch_dset[n, 0] = patch_0
        patch_dset[n, 1] = patch_1
        pt_dset[n] = pt.xyz
        cam_dset[n, 0] = cam_pos[cam_0]
        cam_dset[n, 1] = cam_pos[cam_1]
        n += 1

        print(n, i_house)

        if n == nmax:
            break

        """
        subplot(121)
        imshow(imgs[im_id_0])
        plot(*xy_0, '.')
        subplot(122)
        imshow(imgs[im_id_1])
        plot(*xy_1, '.')

        subplot(121)
        imshow(patch_0)
        plot(128, 128, '.')
        subplot(122)
        imshow(patch_1)
        plot(128, 128, '.')
        """
