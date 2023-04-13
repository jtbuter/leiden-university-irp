import cv2
import numpy as np
import matplotlib.pyplot as plt

from irp import utils

data = utils.make_sample_label("case10_11.png", "case10_10.png")
train_image, train_label = data[0]
test_image, test_label = data[1]

def get_intensity_counts(image):   
    intensities = np.zeros((256,), np.uint8)
    ids, values = np.unique(image, return_counts=True)
    intensities[ids] = values

    return intensities

bit_mask = cv2.inRange(test_image, 0, 62)

f = plt.figure()

f.add_subplot(121)
plt.imshow(np.hstack([test_image, test_label, bit_mask]), cmap='gray')

vals = test_image.flatten()
counts, bins = np.histogram(vals, range(257))

f.add_subplot(122)
plt.bar(bins[:-1] - 0.5, counts, width=1, edgecolor='none')
plt.xlim([-0.5, 255.5])

plt.show()

# get_intensity_counts(train_image)