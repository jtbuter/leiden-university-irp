import cv2
import numpy as np
from gym import spaces

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

def get_contours(mask):
    return cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

def get_compactness(contour, object_area):
    object_perimeter = cv2.arcLength(contour, True)

    return (4 * np.pi * object_area) / (object_perimeter ** 2)

def get_area(contour):
    return cv2.contourArea(contour)

def normalize_area(sub_image, object_area):
    height, width = sub_image.shape
    sub_image_area = height * width

    return (sub_image_area - object_area) / sub_image_area

def get_dims(*args):
    dims = tuple()

    for space in args:
        if isinstance(space, spaces.MultiDiscrete):
            dims += tuple(space.nvec)
        else:
            dims += (space.n,)

    return dims

def discretize(sample, grid):
    return tuple(int(np.digitize(s, g)) for s, g in zip(sample, grid))

def make_simple_train_test(train_name, test_name):
    base_url = "/home/joel/Documents/leiden/introductory_research_project/data/trus/"
    image_url = base_url + "images/" 
    label_url = base_url + "labels/"

    train_test_images, train_test_labels = [], []
    
    for name in [train_name, test_name]:
        image = read_image(image_url + name)
        label = read_image(label_url + name)

        subimages, coords = extract_subimages(image, 32, 16)
        sublabels, coords = extract_subimages(label, 32, 16)

        subimage = subimages[184]
        sublabel = sublabels[184]

        train_test_images.append(subimage)
        train_test_labels.append(sublabel)
    
    return list(zip(train_test_images, train_test_labels))
