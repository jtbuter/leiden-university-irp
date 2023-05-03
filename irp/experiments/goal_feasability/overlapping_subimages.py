import irp
import irp.utils
import irp.envs
import numpy as np
import os
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt

subimage_width, subimage_height = 16, 8
filename = 'case10_10.png'

# Define the paths to the related parent directories
base_path = os.path.join(irp.GIT_DIR, "../data/trus/")
image_path = os.path.join(base_path, 'images')
label_path = os.path.join(base_path, 'labels')

# Read the image and label
image = irp.utils.read_image(os.path.join(image_path, filename))
label = irp.utils.read_image(os.path.join(label_path, filename))

image = median_filter(image, 7)
overlap = 0.5

subimages = np.asarray(irp.utils.extract_subimages(image, subimage_width, subimage_height, overlap=overlap)[0])
sublabels = np.asarray(irp.utils.extract_subimages(label, subimage_width, subimage_height, overlap=overlap)[0])

neighborhood = irp.utils.get_neighborhood((240, 272), image, 16, 8, overlap=overlap, n_size=2)
n_thresholds = 6

a = np.zeros((512, 512))

for neighbor in neighborhood:
    id = irp.utils.coord_to_id(neighbor, image, 16, 8, overlap=overlap)
    subimage = subimages[id]
    sublabel = sublabels[id]

    mini, maxi = np.min(subimage), np.max(subimage)

    tis = np.linspace(mini, maxi, n_thresholds, dtype=np.uint8).tolist()
    tis = np.concatenate(([mini - 1], tis))

    best_dissim = np.inf
    best_bitmask = -1

    for ti in tis:
        bitmask = irp.envs.utils.apply_threshold(subimage, ti)
        dissim = irp.envs.utils.compute_dissimilarity(bitmask, sublabel)

        if dissim < best_dissim:
            best_dissim = dissim
            best_bitmask = bitmask

    print(neighbor, id, best_dissim <= 0.05)
    x, y = neighbor
    # a[y:y+8,x:x+16] = sublabel

    if best_dissim > 0.12:
        plt.title(str(best_dissim))
        plt.imshow(np.hstack([subimage, sublabel, best_bitmask]), cmap='gray')
        plt.show()

# plt.imshow(a)
# plt.show()