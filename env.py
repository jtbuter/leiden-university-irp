import numpy as np
import gymnasium as gym
import cv2
import matplotlib.pyplot as plt

def compute_dissimilarity(mask, sublabel):
    width, height = sublabel.shape

    return np.sum(np.logical_xor(mask, sublabel)) / (width * height)

def threshold_subimage(subimage, threshold):
    return cv2.threshold(subimage, thresh = threshold, maxval = 255, type = cv2.THRESH_BINARY)[-1]


def morph_mask(mask, size):
    if size == 0:
        return mask

    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, se)


def create_threshold_space(subimage, delta_i = 15):
    glmin, glmax = np.min(subimage), np.max(subimage)

    return np.linspace(glmin, glmax, delta_i)

def discretize(sample, grid):
    return list(int(np.digitize(s, g)) for s, g in zip(sample, grid))  # apply along each dimension

class Trus(gym.Env):
    def __init__(self, subimage, sublabel, x_s, y_s, state_grid, delta_i = 15, disk_sizes = [0, 2, 5]):
        self.subimage = subimage
        self.sublabel = sublabel
        self.x_s, self.y_s = x_s, y_s
        self.state_grid = state_grid

        features, bins = state_grid.shape
        bins = bins + 1

        self.actions = [create_threshold_space(self.subimage, delta_i), disk_sizes]
        self.action_space = gym.spaces.MultiDiscrete([delta_i, len(disk_sizes)])
        self.observation_space = gym.spaces.MultiDiscrete([bins] * features)
        self.dsim_old = np.inf

        print(self.action_space.shape)

    def observation(self):
        height, width = self.subimage.shape
        subimage_area = height * width
        contours = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        N_o = len(contours)

        # No objects were found
        if N_o == 0:
            return discretize((0, 0, self.x_s, self.y_s, 0), self.state_grid)

        c = max(contours, key = cv2.contourArea)

        object_area = cv2.contourArea(c)

        if object_area == 0:
            compactness = 0
        else:
            object_perimeter = cv2.arcLength(c, True)
            compactness = (4 * np.pi * object_area) / (object_perimeter ** 2)

        area = (subimage_area - object_area) / subimage_area

        return discretize((area, compactness, self.x_s, self.y_s, N_o), self.state_grid)
    

    def reset(self, seed = None, options = None):
        super().reset(seed = seed)

        self.dsim_old = np.inf
        self.mask = np.zeros_like(self.subimage)

        return self.observation(), {}


    def step(self, action):
        thresh_index, size_index = action
        threshold, size = self.actions[0][thresh_index], self.actions[1][size_index]

        mask = threshold_subimage(self.subimage, threshold)

        self.mask = morph_mask(mask, size)
        self.state = self.observation()

        dsim_new = compute_dissimilarity(self.mask, self.sublabel)

        reward = 10 if dsim_new < self.dsim_old else 0

        self.dsim_old = dsim_new

        return self.state, reward, dsim_new < 0.05, False, {}