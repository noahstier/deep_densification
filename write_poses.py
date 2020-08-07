import importlib
import glob
import os

import config
import numpy as np
import scipy.spatial

spec = importlib.util.spec_from_file_location(
    "colmap_reader", config.colmap_reader_script
)
colmap_reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(colmap_reader)

house_dirs = glob.glob(
    os.path.join(
        "/home/noah/Documents/scenecompletion/deep_densification/suncg_output", "*",
    )
)


for house_dir in tqdm.tqdm(house_dirs):
    sfm_imfile = os.path.join(house_dir, "sfm/sparse/auto/images.bin")
    ims = colmap_reader.read_images_binary(sfm_imfile)

    im_ids = sorted(ims.keys())

    intr_file = os.path.join(house_dir, "camera_intrinsics")
    camera_intrinsic = np.loadtxt(intr_file)[0].reshape(3, 3)
    camera_extrinsics = np.zeros((len(im_ids), 4, 4))
    for i, im_id in enumerate(im_ids):
        im = ims[im_id]
        qw, qx, qy, qz = im.qvec
        r = scipy.spatial.transform.Rotation.from_quat([qx, qy, qz, qw]).as_matrix().T
        t = np.linalg.inv(-r.T) @ im.tvec
        camera_extrinsics[i] = np.array(
            [[*r[0], t[0]], [*r[1], t[1]], [*r[2], t[2]], [0, 0, 0, 1]]
        )

    np.savez(
        os.path.join(house_dir, "poses.npz"),
        intrinsic=camera_intrinsic,
        extrinsics=camera_extrinsics,
    )
