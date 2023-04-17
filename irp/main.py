from scipy.ndimage import median_filter

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

import irp
import irp.utils

def get_intensity_counts(image):   
    intensities = np.zeros((256,), np.uint8)
    ids, values = np.unique(image, return_counts=True)
    intensities[ids] = values

    return intensities

label = irp.utils.read_image(os.path.join(irp.ROOT_DIR, '../../data/trus/labels/case10_10.png'))
image = irp.utils.read_image(os.path.join(irp.ROOT_DIR, '../../data/trus/images/case10_10.png'))
subimage = irp.utils.extract_subimages(image, 32, 16)[0][184]
sublabel = irp.utils.extract_subimages(label, 32, 16)[0][184]

filtered_image = median_filter(image, 7)
filtered_subimage = irp.utils.extract_subimages(filtered_image, 32, 16)[0][184]

thresholds = np.linspace(np.min(filtered_subimage), np.max(filtered_subimage), 30, dtype=np.uint8)

for ti in thresholds:
    bit_mask = cv2.threshold(filtered_subimage, int(ti), 255, cv2.THRESH_BINARY_INV)[1]

    fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(15, 15))

    ax0.imshow(filtered_subimage, cmap='gray')
    ax1.imshow(bit_mask, cmap='gray')
    ax2.imshow(sublabel, cmap='gray')

    plt.show()

# vals = subimage.flatten()
# counts, bins = np.histogram(vals, range(257))

# plt.bar(bins[:-1] - 0.5, counts, width=1, edgecolor='none')
# plt.xlim([-0.5, 255.5])
# plt.show()

# vals = filtered_subimage.flatten()
# counts, bins = np.histogram(vals, range(257))

# plt.bar(bins[:-1] - 0.5, counts, width=1, edgecolor='none')
# plt.xlim([-0.5, 255.5])
# plt.show()

# bit_mask = cv2.inRange(test_image, 0, 62)

# f = plt.figure()

# f.add_subplot(121)
# plt.imshow(np.hstack([test_image, test_label, bit_mask]), cmap='gray')

# vals = test_image.flatten()
# counts, bins = np.histogram(vals, range(257))

# f.add_subplot(122)
# plt.bar(bins[:-1] - 0.5, counts, width=1, edgecolor='none')
# plt.xlim([-0.5, 255.5])

# plt.show()

# # get_intensity_counts(train_image)