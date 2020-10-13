import cv2

uvmap = imread("fuze_uv.jpg.bak").copy()
mask = np.all(uvmap == 127, axis=-1)

img = imread("pattern.jpg")


img = cv2.resize(img, (uvmap.shape[1], uvmap.shape[0]))

uvmap[mask] = img[mask]

cv2.imwrite("fuze_uv.jpg", cv2.cvtColor(uvmap, cv2.COLOR_BGR2RGB))
