from __future__ import annotations
import os
import typing
from typing import Union, Optional, Callable, Dict, Any, Tuple, List
import irp

import cv2
import numpy as np
import matplotlib.pyplot as plt

import gym
from gym import spaces
from gym.wrappers import TimeLimit

from scipy.ndimage import median_filter

import irp.wrappers.discretize
import irp.envs.utils
# import irp.envs.sahba.sahba_2008_env

if typing.TYPE_CHECKING:
    from irp.q import Q

def apply_threshold(sample, *tis):
    if len(tis) == 1:
        bit_mask = cv2.threshold(sample, int(tis[0]), 255, cv2.THRESH_BINARY_INV)[1]
    else:
        bit_mask = cv2.inRange(sample, int(tis[0]), int(tis[1]))

    return bit_mask

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

def make_sample_label(*file_names, idx=184):
    base_path = os.path.join(irp.ROOT_DIR, "../../data/trus/")
    image_path = os.path.join(base_path, 'images')
    label_path = os.path.join(base_path, 'labels')

    images, labels = [], []
    
    for file_name in file_names:
        image = read_image(os.path.join(image_path, file_name))
        image = median_filter(image, 7)
        label = read_image(os.path.join(label_path, file_name))

        subimages, coords = extract_subimages(image, 16, 8)
        sublabels, coords = extract_subimages(label, 16, 8)

        subimage = subimages[idx]
        sublabel = sublabels[idx]

        images.append(subimage)
        labels.append(sublabel)
    
    return list(zip(images, labels))

def get_subimages(filename):
    # Define the paths to the related parent directories
    base_path = os.path.join(irp.GIT_DIR, "../data/trus/")
    image_path = os.path.join(base_path, 'images')
    label_path = os.path.join(base_path, 'labels')
    # Read the image and label
    image = read_image(os.path.join(image_path, filename))
    image = median_filter(image, 7)
    label = read_image(os.path.join(label_path, filename))

    subimages = extract_subimages(image, 32, 16)[0]
    sublabels = extract_subimages(label, 32, 16)[0]

    return subimages, sublabels

def show(sample):
    plt.imshow(sample, vmin=0, vmax=255, cmap='gray')
    plt.show()

def get_best_dissimilarity(
    sample: np.ndarray,
    label: np.ndarray,
    n_thresholds: int
)-> np.float64:
    best = np.inf
    tis = irp.envs.utils.get_intensity_spectrum(sample, n_thresholds)

    for ti in tis:
        bitmask = irp.envs.utils.apply_threshold(sample, ti)
        dissim = irp.envs.utils.compute_dissimilarity(bitmask, label)

        if dissim < best:
            best = dissim

    return best