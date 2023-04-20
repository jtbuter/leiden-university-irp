from abc import abstractmethod
import gym
import numpy as np
import cv2
import matplotlib.pyplot as plt

import irp.utils

class UltraSoundEnv(gym.Env):
    # Has to be specified to pass the stable-baselines .render() check
    metadata = {
        "render.modes": ['human'], "render_fps": 1
    }

    # Action map which allows for modifying the threshold index
    action_map = np.array([
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 0), (0, 1),
        (1, -1), (1, 0), (1, 1)
    ])

    def __init__(self, sample = None, label = None, num_thresholds = 10, render_mode=None):
        self.label = label.copy()
        self.sample = sample.copy()
        self.state = None

        self.observation_space = gym.spaces.Box(0, np.inf, (3,))
        self.action_space = gym.spaces.Discrete(n=len(self.action_map))

        self.num_thresholds = num_thresholds
        self.threshold_ids = np.array([0, self.num_thresholds - 1])
        self.thresholds = np.linspace(np.min(sample), np.max(sample), num_thresholds, dtype=np.uint8)

        self.old_dissim = None
        self.render_mode = render_mode

    def reward(self, *args):
        return self._reward(*args)

    def _reward(self, dissim):
        if dissim < self.old_dissim:
            return 10
        elif dissim == self.old_dissim:
            return 0
        elif dissim > self.old_dissim:
            return 0

    def _is_valid_action(self, lt, rt):
        return lt <= rt

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def reset(self):
        pass

    def observation(self, bit_mask):
        contours = irp.utils.get_contours(bit_mask)
        num_objects = len(contours)

        if num_objects == 0:
            return (1., 0., 0.)

        # Get the biggest object based on its area
        biggest_object = max(contours, key=cv2.contourArea)
        object_area = irp.utils.get_area(biggest_object)

        if object_area == 0:
            return (1., 0., num_objects)

        compactness = irp.utils.get_compactness(biggest_object, object_area)
        normalized_area = irp.utils.normalize_area(bit_mask, object_area)

        return (normalized_area, compactness, num_objects)

    def render(self, mode = None):
        self._render()

    def _render(self):
        # Threshold ids obtained after resetting or performing a step
        lt, rt = self.thresholds[self.threshold_ids]

        # Apply the thresholds
        bit_mask = cv2.inRange(self.sample, int(lt), int(rt))
        state = cv2.bitwise_and(self.sample, self.sample, mask=bit_mask)

        # Show the final result
        plt.imshow(np.hstack([self.label, state]), cmap='gray', vmin=0, vmax=1)
        plt.show()

    def _apply_opening(self, bit_mask, size):
        # Check that the structuring element has a size
        if size == 0:
            return bit_mask

        # Apply an opening to the bit-mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        bit_mask = cv2.morphologyEx(bit_mask, cv2.MORPH_OPEN, kernel)

        return bit_mask

    def close(self):
        pass
