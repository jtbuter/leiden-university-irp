from stable_baselines3.common import env_checker
import gym
import numpy as np
import cv2
import matplotlib.pyplot as plt

import irp.utils

class UltraSoundEnv(gym.Env):
    metadata = {
        "render.modes": ['human'], "render_fps": 1
    }
    action_map = np.array([(-1, -1), (-1, 1), (1, -1), (1, 1)])

    def __init__(self, sample = None, label = None, num_thresholds = 10, render_mode=None):
        super(UltraSoundEnv, self).__init__()

        self.label = label.copy()
        self.sample = sample.copy()
        self.state = None

        self.action_space = gym.spaces.Discrete(n = 4)
        self.observation_space = gym.spaces.Box(0, 255, self.sample.shape + (1,), np.uint8)

        self.num_thresholds = num_thresholds
        self.threshold_ids = np.array([0, self.num_thresholds - 1])
        self.thresholds = np.linspace(np.min(sample), np.max(sample), num_thresholds, dtype = np.uint8)

        self.old_dissim = None
        self.render_mode = render_mode

    def _get_reward(self, dissim):
        if dissim < self.old_dissim:
            return 10
        elif dissim == self.old_dissim:
            return 0
        elif dissim > self.old_dissim:
            return 0

    def step(self, action):
        new_threshold_ids = irp.utils.process_thresholds(action, self.action_map, self.threshold_ids, self.num_thresholds)
        lt, rt = self.thresholds[new_threshold_ids]

        bit_mask = cv2.inRange(self.sample, int(lt), int(rt))
        next_state = cv2.bitwise_and(self.sample, self.sample, mask = bit_mask)

        dissim = irp.utils.compute_dissimilarity(bit_mask, self.label)
        reward = self._get_reward(dissim)
        is_done = bool(dissim < 0.05)

        self.old_dissim = dissim
        self.threshold_ids = new_threshold_ids
        self.state = next_state.reshape(*next_state.shape, 1)

        return self.state, reward, is_done, {}

    def reset(self):
        bit_mask = np.full(self.label.shape, 255)
        self.state = self.sample.reshape(*self.sample.shape, 1)
        self.threshold_ids = np.array([0, self.num_thresholds - 1])
        self.old_dissim = irp.utils.compute_dissimilarity(bit_mask, self.label)

        return self.state

    def render(self, mode = None):
        # Threshold ids obtained after resetting or performing a step
        lt, rt = self.thresholds[self.threshold_ids]

        # Apply the thresholds
        bit_mask = cv2.inRange(self.sample, int(lt), int(rt))
        state = cv2.bitwise_and(self.sample, self.sample, mask = bit_mask)

        # Show the final result
        plt.imshow(np.hstack([self.label, state]), cmap = 'gray', vmin = 0, vmax = 1)
        plt.show()

    def close(self):
        pass
