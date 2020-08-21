import glob
import multiprocessing as mp
import os
import pickle

import numpy as np
import PIL.Image
import tqdm


def work(tup):
    house_name, img_name, cat_imgfile = tup
    img_name = os.path.basename(cat_imgfile)
    inds, counts = np.unique(PIL.Image.open(cat_imgfile), return_counts=True)
    return house_name, img_name, {i: c for (i, c) in zip(inds, counts)}


pool = mp.Pool(10)
q = []

per_img_classes = {}

house_dirs = sorted(glob.glob("/home/noah/data/suncg_output/*"))
for house_dir in tqdm.tqdm(house_dirs):
    house_name = os.path.basename(house_dir)
    per_img_classes[house_name] = {}
    cat_imgfiles = sorted(glob.glob(os.path.join(house_dir, "imgs/category/*.png")))
    for cat_imgfile in cat_imgfiles:
        img_name = os.path.basename(cat_imgfile)
        q.append((house_name, img_name, cat_imgfile))

outq = list(tqdm.tqdm(pool.imap(work, q), total=len(q)))

for house_name, img_name, d in outq:
    per_img_classes[house_name][img_name] = d

with open("per_img_classes.pkl", "wb") as f:
    pickle.dump(per_img_classes, f)

with open("per_img_classes.pkl", "rb") as f:
    per_img_classes = pickle.load(f)

imgs_w_chair = []

house_dirs = sorted(glob.glob("/home/noah/data/suncg_output/*"))
for house_dir in tqdm.tqdm(house_dirs):
    house_name = os.path.basename(house_dir)
    cat_imgfiles = sorted(glob.glob(os.path.join(house_dir, "imgs/category/*.png")))
    for cat_imgfile in cat_imgfiles:
        img_name = os.path.basename(cat_imgfile)
        if 11 in per_img_classes[house_name][img_name]:
            imgs_w_chair.append(cat_imgfile)


import glob
import os
import importlib
import multiprocessing as mp
import pickle

import PIL.Image
import tqdm

colmap_reader_script = "/home/noah/Documents/colmap/scripts/python/read_write_model.py"
spec = importlib.util.spec_from_file_location("colmap_reader", colmap_reader_script)
colmap_reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(colmap_reader)

house_dirs = sorted(glob.glob("/home/noah/data/suncg_output/*"))
q = []
for house_dir in house_dirs:
    house_name = os.path.basename(house_dir)
    imfile = os.path.join(house_dir, "sfm/sparse/auto/images.bin")
    q.append((house_dir, house_name, imfile))


def work(tup):
    house_dir, house_name, imfile = tup

    ims = colmap_reader.read_images_binary(imfile)
    d = {}
    for im in ims.values():
        pt_inds = np.argwhere(im.point3D_ids != -1).flatten()
        xys = im.xys[pt_inds]
        pt_ids = im.point3D_ids[pt_inds]
        uv_inds = np.floor(xys).astype(int)
        cat_imgfile = os.path.join(
            house_dir, "imgs/category", im.name.replace("jpg", "png")
        )
        cat_img = np.asarray(
            PIL.Image.open(cat_imgfile).transpose(PIL.Image.FLIP_TOP_BOTTOM)
        )
        cats = cat_img[uv_inds[:, 1], uv_inds[:, 0]]

        for (pt_id, cat) in zip(pt_ids, cats):
            if pt_id not in d:
                d[pt_id] = []
            d[pt_id].append(cat)

    return house_name, d


pool = mp.Pool(12)
outq = list(tqdm.tqdm(pool.imap(work, q), total=len(q)))

pt_classes = {house_dir: d for (house_dir, d) in outq}

for house_dir, d in tqdm.tqdm(pt_classes.items()):
    house_name = os.path.basename(house_dir)
    with open("pt_classes/" + house_name + ".pkl", "wb") as f:
        pickle.dump(d, f)
