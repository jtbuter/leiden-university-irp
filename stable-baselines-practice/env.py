import gym
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import utils

class UltraSoundEnv(gym.Env):
    action_map = np.array([(-1, -1), (-1, 1), (1, -1), (1, 1)])

    def __init__(self, sample = None, label = None, num_thresholds = 10):
        super(UltraSoundEnv, self).__init__()

        self.label = label.copy()
        self.sample = sample.copy()
        self.state = self.sample.reshape(*self.sample.shape, 1)

        self.action_space = gym.spaces.Discrete(n = 4)
        self.observation_space = gym.spaces.Box(0, 255, self.state.shape, np.uint8)

        self.num_thresholds = num_thresholds
        self.threshold_ids = np.array([0, self.num_thresholds - 1])
        self.thresholds = np.linspace(np.min(sample), np.max(sample), num_thresholds, dtype = np.uint8)

        bit_mask = np.full(self.label.shape, 255)
        self._old_dissim = utils.compute_dissimilarity(bit_mask, self.label)


    def _reward(self, dissim):
        if dissim < self._old_dissim:
            return 10
        elif dissim == self._old_dissim:
            return 0
        elif dissim > self._old_dissim:
            return 0


    def step(self, action):
        new_threshold_ids = utils.process_thresholds(action, self.action_map, self.threshold_ids, self.num_thresholds)
        lt, rt = self.thresholds[new_threshold_ids]

        bit_mask = cv.inRange(self.sample, int(lt), int(rt))
        next_state = cv.bitwise_and(self.sample, self.sample, mask = bit_mask)

        dissim = utils.compute_dissimilarity(bit_mask, self.label)
        reward = self._reward(dissim)
        is_done = bool(dissim < 0.05)

        self._old_dissim = dissim
        self.threshold_ids = new_threshold_ids
        self.state = next_state.reshape(*next_state.shape, 1)

        return self.state, reward, is_done, {}


    def reset(self):
        bit_mask = np.full(self.label.shape, 255)
        self.state = self.sample.reshape(*self.sample.shape, 1)
        self.threshold_ids = np.array([0, self.num_thresholds - 1])
        self._old_dissim = utils.compute_dissimilarity(bit_mask, self.label)

        return self.state


    def render(self):
        plt.imshow(np.squeeze(self.state), cmap = 'gray', vmin = 0, vmax = 1)
        plt.show()


    def close(self):
        pass


class BinaryUltraSoundEnv(UltraSoundEnv):
    def __init__(self, sample = None, label = None, num_thresholds = 10):
        super().__init__(sample, label, num_thresholds)

        self.observation_space = gym.spaces.MultiDiscrete((35, 35, 35))

    
