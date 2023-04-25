import cv2
import numpy as np
import matplotlib.pyplot as plt

from irp import utils

train_path = 'case10_12.png'
test_path = 'case10_11.png'

(train_x, train_y), (test_x, test_y) = utils.make_sample_label(train_path, test_path)

all_tis = [
    np.linspace(np.min(train_x), np.max(train_x), 15),
    np.linspace(np.min(test_x), np.max(test_x), 15)
]

names = ['train', 'test']

for i, obj in enumerate([(train_x, train_y), (test_x, test_y)]):
    x, y = obj
    tis = all_tis[i]
    best_dissim = np.inf
    best_bit_mask = None

    for ti_left in tis:
        for ti_right in tis:
            bit_mask = cv2.inRange(x, int(ti_left), int(ti_right))
            dissim = utils.compute_dissimilarity(bit_mask, y)

            if dissim < best_dissim:
                best_dissim = dissim
                best_bit_mask = bit_mask

    plt.title(str(best_dissim))
    plt.imshow(np.hstack([y, best_bit_mask]), cmap='gray', vmin=0, vmax=1)
    plt.show()

print(utils.compute_dissimilarity(best_bit_mask, test_y))
