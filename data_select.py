import glob
import os

import numpy as np
import PIL.Image
import tqdm


pixel_counts = np.zeros(90)

cat_imgfiles = sorted(glob.glob("/home/noah/data/suncg_output/*/imgs/category/*.png"))
for cat_imgfile in tqdm.tqdm(cat_imgfiles):
    cat_img = PIL.Image.open(cat_imgfile)
    inds, counts = np.unique(cat_img, return_counts=True)
    pixel_counts[inds] += counts
