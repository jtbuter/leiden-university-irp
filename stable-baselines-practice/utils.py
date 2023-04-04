import numpy as np
import cv2

def process_thresholds(action, action_map, tis, n_thresholds):
    return np.clip(tis + action_map[action], 0, n_thresholds - 1)


def compute_dissimilarity(bit_mask, label):
    height, width = label.shape

    return np.sum(np.logical_xor(bit_mask, label)) / (height * width)


def read_image(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def extract_subimages(image, subimage_width, subimage_height):
    height, width = image.shape
    subimages, coords = [], []

    for y in range(0, height, subimage_height):
        for x in range(0, width, subimage_width):
            subimage = image[y:y + subimage_height, x:x + subimage_width]

            subimages.append(subimage)
            coords.append((x, y))

    return subimages, coords