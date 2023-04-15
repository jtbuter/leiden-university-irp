import typing
from typing import Union, Optional, Callable, Dict, Any, Tuple, List

from stable_baselines3.common.vec_env import (VecEnv, DummyVecEnv)
from stable_baselines3.common.monitor import Monitor

import cv2
import numpy as np

import gym
from gym import spaces
from gym.wrappers import TimeLimit

from scipy.ndimage import median_filter

from irp.wrappers import Discretize
from irp.envs import Sahba2008UltraSoundEnv

if typing.TYPE_CHECKING:
    from irp.q import Q  # pytype: disable=pyi-error

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

def make_sample_label(*args):
    base_url = "/home/joel/Documents/leiden/introductory_research_project/data/trus/"
    image_url = base_url + "images/" 
    label_url = base_url + "labels/"

    train_test_images, train_test_labels = [], []
    
    for name in args:
        image = median_filter(read_image(image_url + name), 7)
        label = read_image(label_url + name)

        subimages, coords = extract_subimages(image, 32, 16)
        sublabels, coords = extract_subimages(label, 32, 16)

        subimage = subimages[184]
        sublabel = sublabels[184]

        train_test_images.append(subimage)
        train_test_labels.append(sublabel)
    
    return list(zip(train_test_images, train_test_labels))

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
    model: 'Q', env: gym.Env, n_eval_episodes: int = 10, n_eval_timesteps: float = np.inf
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    episode_rewards = []

    episode_count = 0
    step_count = 0
    current_reward = 0

    observation = env.reset()[0]

    while episode_count < n_eval_episodes:
        action = model.predict(observation, deterministic=True)

        next_state, reward, done, info = env.step([action])
        next_state, reward, done, info = unwrap_sb3_env(next_state, reward, done, info)

        current_reward += reward
        step_count += 1

        if done:
            episode_rewards.append(current_reward)

            episode_count += 1
            current_reward = 0

        observation = next_state

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward

def setup_environment(
    image: np.ndarray, label: np.ndarray, num_thresholds: int,
    vjs: tuple, lows: dict, highs: dict, bins: tuple, episode_length: int
) -> gym.Env:
    # Initialize the environment
    env = Sahba2008UltraSoundEnv(image, label, num_thresholds, vjs)

    # Cast continuous values to bins
    env = Discretize(env, lows, highs, bins)
    
    # Set a maximum episode length
    env = TimeLimit(env, episode_length)

    return env
