from __future__ import annotations
import os
import itertools
import typing
from typing import Union, Optional, Callable, Dict, Any, Tuple, List
import irp

import diplib as dip

import copy
import cv2
import numpy as np
import matplotlib.pyplot as plt

import gym
from gym import spaces
from gym.wrappers import TimeLimit

from scipy.ndimage import median_filter

import sklearn.metrics

import irp.wrappers.discretize
import irp.envs as envs

if typing.TYPE_CHECKING:
    from irp.q import Q

def process_thresholds(action, action_map, tis, n_thresholds):
    return np.clip(tis + action_map[action], 0, n_thresholds - 1)

def compute_dissimilarity(bit_mask, label):
    height, width = label.shape

    return np.sum(np.logical_xor(bit_mask, label)) / (height * width)

def read_image(path, add_root: Optional[bool] = False):
    if add_root:
        # Define the paths to the related parent directories
        base_path = os.path.join(irp.GIT_DIR, "../data/trus/")
        image_path = os.path.join(base_path, 'images')
        label_path = os.path.join(base_path, 'labels')
        # Read the image and label
        image_path = os.path.join(image_path, path)
        label_path = os.path.join(label_path, path)
        
        return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def extract_subimages(image, subimage_width, subimage_height, overlap=0):
    if isinstance(image, tuple):
        image = np.zeros(image)

    height, width = image.shape
    subimages, coords = [], []

    height_step_size = int(subimage_height * (1 - overlap))
    width_step_size = int(subimage_width * (1 - overlap))
    sizes = []

    for y in range(0, height - (subimage_height - height_step_size), height_step_size):
        for x in range(0, width - (subimage_width - width_step_size), width_step_size):
            subimage = image[y:y + subimage_height, x:x + subimage_width]

            sizes.append(subimage.size)

            subimages.append(subimage)
            coords.append((x, y))

    assert len(set(sizes)) == 1, f"Subimages have differing sizes: {set(sizes)}"

    return subimages, coords

def get_contours(mask):
    return cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

def get_largest_component(bit_mask):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bit_mask, 8, cv2.CV_32S
    )

    # We've only found background pixels
    if num_labels == 1:
        return num_labels, 0, bit_mask

    # Get label of largest connected component, skipping the background
    idx = np.argmax(stats[1:,4]) + 1
    # Retrieve the area of the largest component
    area = stats[idx][4]
    # Select the labels that correspond to the largest component label
    largest_component = (labels == idx).astype(np.uint8) * 255

    return num_labels, area, largest_component

def get_compactness(contour, object_area):
    object_perimeter = cv2.arcLength(contour, True)

    return (4 * np.pi * object_area) / (object_perimeter ** 2)

def get_area(contour):
    # return cv2.countNonZero(contour)
    return cv2.contourArea(contour)

def normalize_area(sub_image, object_area):
    height, width = sub_image.shape
    sub_image_area = height * width

    return object_area / sub_image_area

def get_dims(*args):
    dims = tuple()

    for space in args:
        if isinstance(space, spaces.MultiDiscrete):
            dims += tuple(space.nvec)
        else:
            dims += (space.n,)

    return dims

def discrete(sample, grid) -> Tuple[int, ...]:
    # TODO: Checken of searchsorted en digitize dezelfde resultaten geven
    # return tuple(int(np.digitize(s, g)) for s, g in zip(sample, grid))
    return tuple(int(np.searchsorted(g, s)) for s, g in zip(sample, grid))

def make_sample_label(*file_names, idx=184, width=32, height=16, overlap=0, ellipse=[1,1]):
    base_path = os.path.join(irp.ROOT_DIR, "../../data/trus/")
    image_path = os.path.join(base_path, 'images')
    label_path = os.path.join(base_path, 'labels')

    images, labels = [], []
    
    for file_name in file_names:
        image = read_image(os.path.join(image_path, file_name))
        image = median_filter(image, 15)
        label = read_image(os.path.join(label_path, file_name))

        subimages, coords = extract_subimages(image, width, height, overlap)
        sublabels, coords = extract_subimages(label, width, height, overlap)

        if idx is None:
            images.append(subimages)
            labels.append(sublabels)
        else:
            subimage = subimages[idx]
            sublabel = sublabels[idx]

            images.append(subimage)
            labels.append(sublabel)
    
    return np.asarray(list(zip(images, labels)))

def str_to_builtin(value: str, builtin: str = None):
    # TODO: It's safer to use this way of casting variables, but for now use
    # eval()
    # return locate(builtin)(value)

    return eval(value)

def params_to_modelname(**kwargs):
    return ','.join(list(f'{key}={value}' for key, value in kwargs.items()))

def unwrap_sb3_env(*args):
    return tuple(arg[0] for arg in args)

def evaluate_policy(
    model: Q, env: gym.Env, n_eval_episodes: int = 10, n_eval_timesteps: float = np.inf
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:

    # These packages are extremely slow, so delay loading them
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    episode_dissims = []

    episode_count = 0
    steps = 0

    observation = env.reset()[0]

    # Keep taking steps until an episode is finished
    while episode_count < n_eval_episodes:
        action = model.predict(observation, deterministic=True)

        next_state, reward, done, info = env.step([action])
        next_state, reward, done, info = unwrap_sb3_env(next_state, reward, done, info)
        
        steps += 1

        truncated = steps >= n_eval_timesteps or 'TimeLimit.truncated' in info

        # TODO: Optionally use `done`, we don't use this right now, as the environment
        # shouldn't have access to the ground-truth in evaluation mode.
        if truncated:
            episode_dissims.append(info['dissimilarity'])
            steps = 0
            episode_count += 1

            if truncated:
                next_state = env.reset()[0]

        observation = next_state

    mean_reward = np.mean(episode_dissims)
    std_reward = np.std(episode_dissims)

    return mean_reward, std_reward

# def setup_environment(
#     image: np.ndarray, label: np.ndarray, num_thresholds: int,
#     vjs: tuple, lows: dict, highs: dict, bins: tuple, episode_length: int,
#     env_cls = None
# ) -> gym.Env:
#     # Initialize the environment
#     if env_cls is None:
#         env = irp.envs.sahba.sahba_2008_env.Sahba2008UltraSoundEnv(image, label, num_thresholds, vjs)
#     else:
#         env = env_cls(image, label, num_thresholds, vjs)

#     # Cast continuous values to bins
#     env = irp.wrappers.discretize.Discretize(env, lows, highs, bins)
    
#     # Set a maximum episode length
#     env = TimeLimit(env, episode_length)

#     return env

def parse_highs(area, compactness, objects, label):
    height, width = label.shape

    if objects == "normalize":
        objects = int(np.ceil(width / 2) * np.ceil(height / 2))

    return {
        'area': area,
        'compactness': compactness,
        'objects': objects
    }

def get_subimages(filename, width=32, height=16, overlap=0):
    # Define the paths to the related parent directories
    base_path = os.path.join(irp.GIT_DIR, "../data/trus/")
    image_path = os.path.join(base_path, 'images')
    label_path = os.path.join(base_path, 'labels')
    # Read the image and label
    image = read_image(os.path.join(image_path, filename))
    image = median_filter(image, 15)
    label = read_image(os.path.join(label_path, filename))

    subimages = extract_subimages(image, width, height, overlap)[0]
    sublabels = extract_subimages(label, width, height, overlap)[0]

    return subimages, sublabels

def id_to_coord(
    id: int,
    shape: Tuple[int, int],
    subimage_width: int,
    subimage_height: int,
    overlap: Optional[float] = 0
) -> Tuple[int, int]:
    height, width = shape
    i = 0

    height_step_size = int(subimage_height * (1 - overlap))
    width_step_size = int(subimage_width * (1 - overlap))

    for y in range(0, height - (subimage_height - height_step_size), height_step_size):
        for x in range(0, width - (subimage_width - width_step_size), width_step_size):
            if i == id:
                return (x, y)

            i += 1

def coord_to_id(
    coord: Tuple[int, int],
    shape: Tuple[int, int],
    subimage_width: int,
    subimage_height: int,
    overlap: Optional[float] = 0
) -> int:
    height, width = shape
    i = 0

    height_step_size = int(subimage_height * (1 - overlap))
    width_step_size = int(subimage_width * (1 - overlap))

    for y in range(0, height - (subimage_height - height_step_size), height_step_size):
        for x in range(0, width - (subimage_width - width_step_size), width_step_size):
            if (x, y) == coord:
                return i

            i += 1

def unravel_index(
    i: int,
    width: int,
    height: int,
    divisor: Optional[int] = 512
):
    x, y = (i * width) % divisor, ((i * width) // divisor) * height

    return x, y

def diamond(n):
    a = np.arange(n)
    b = np.minimum(a, a[::-1])
    
    return (b[:, None] + b) >= (n - 1) // 2

def get_neighborhood(
    coord: Union[int, Tuple],
    shape: Tuple[int, int],
    subimage_width: int,
    subimage_height: int,
    overlap: Optional[float] = 0,
    n_size: Optional[int] = 1,
    neighborhood = 'moore'
) -> List[Tuple]:
    if isinstance(coord, int):
        coord = id_to_coord(coord, shape, subimage_width, subimage_height, overlap)

    width_step_size = round((1 - overlap) * subimage_width, 0)
    height_step_size = round((1 - overlap) * subimage_height, 0)

    x, y = coord
    coords = []

    if isinstance(neighborhood, np.ndarray):
        neighbor_map = neighborhood.flatten()
    elif neighborhood == 'neumann':
        neighbor_map = diamond(n_size * 2 + 1).flatten()
    elif neighborhood == 'moore':
        neighbor_map = np.ones((n_size * 2 + 1, n_size * 2 + 1), dtype=bool).flatten()

    for y_i in range(-n_size, n_size + 1):
        y_i *= height_step_size

        for x_i in range(-n_size, n_size + 1):
            x_i *= width_step_size

            coords.append((x + x_i, y + y_i))

    return list(map(tuple, np.asarray(coords)[neighbor_map]))

def get_neighborhood_images(
    subimages: List[np.ndarray],
    sublabels: List[np.ndarray],
    coord: Union[int, Tuple],
    subimage_width: int,
    subimage_height: int,
    overlap: Optional[float] = 0,
    n_size: Optional[int] = 1,
    shape: Optional[Tuple[int, int]] = (512, 512),
    neighborhood: Optional[Union[str, np.ndarray]] = 'moore'
) -> np.ndarray:
    neighborhood_coords = get_neighborhood(coord, shape, subimage_width, subimage_height, overlap, n_size, neighborhood)

    neighborhood_ids = [
        coord_to_id(coord_, shape, subimage_width, subimage_height, overlap) for coord_ in neighborhood_coords
    ]

    return np.asarray([subimages[i] for i in neighborhood_ids]), np.asarray([sublabels[i] for i in neighborhood_ids])

def get_best_dissimilarity(
    subimage,
    sublabel,
    actions: List[Union[int, str]],
    fns: List[Callable],
    return_seq = False
) -> Union[float, Tuple[float, int]]:
    best_dissim = np.inf
    best_sequence = None

    for sequence in itertools.product(*actions):
        fn, action = fns[0], sequence[0]
        actions_ = []

        if not isinstance(action, tuple):
            action = [action]

        actions_.append(action)

        bitmask = fn(subimage, *action)

        for i in range(1, len(fns)):
            fn, action = fns[i], sequence[i] 

            if not isinstance(action, tuple):
                action = [action]
            
            actions_.append(action)

            bitmask = fn(bitmask, *action)

        dissim = envs.utils.compute_dissimilarity(bitmask, sublabel)

        if dissim < best_dissim:
            best_dissim = dissim
            best_sequence = actions_

    if return_seq:
        return float(best_dissim), best_sequence

    return float(best_dissim)

def show(sample):
    plt.imshow(sample, vmin=0, vmax=255, cmap='gray')
    plt.show()

def area_of_overlap(
    label: np.ndarray,
    bitmask: np.ndarray
) -> float:
    tp = ((label == 255) & (bitmask == 255)).sum()
    fn = ((bitmask == 0) & (label != bitmask)).sum()

    if tp + fn == 0:
        return 0.0

    return tp / (tp + fn)

def precision(
    label: np.ndarray,
    bitmask: np.ndarray
) -> float:
    tp = ((label == 255) & (bitmask == 255)).sum()
    fp = ((bitmask == 255) & (label != bitmask)).sum()

    if tp + fp == 0:
        return 0.0

    return tp / (tp + fp)

def f1(
    label: np.ndarray,
    bitmask: np.ndarray
) -> float:
    label = (label / 255).astype(int)
    bitmask = (bitmask / 255).astype(int)

    return sklearn.metrics.f1_score(label.flatten(), bitmask.flatten())

def jaccard(
    label: np.ndarray,
    bitmask: np.ndarray
) -> float:
    intersection = np.logical_and(label, bitmask)
    union = np.logical_or(label, bitmask)

    return np.sum(intersection) / np.sum(union)

def dice(
    label: np.ndarray,
    bitmask: np.ndarray
) -> float:
    tp = ((label == 255) & (bitmask == 255)).sum()
    fp = ((bitmask == 255) & (label != bitmask)).sum()
    fn = ((bitmask == 0) & (label != bitmask)).sum()

    return (2 * tp) / (2 * tp + fp + fn)
    # union = 

    # return (2 * (label & bitmask) / ((label > 0).sum() + (bitmask > 0).sum())