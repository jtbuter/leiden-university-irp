import gymnasium as gym
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

class UltraSoundEnv(gym.Env):
    action_map = np.array([(-1, -1), (-1, 1), (1, -1), (1, 1)])

    def __init__(self, sample, label, n_thresholds = 10):
        self.state = sample.copy()
        self.label = label.copy()
        self.sample = sample.copy()
        self.dissim = self._compute_reward(np.full(sample.shape, 255))
        self.action_space = gym.spaces.Discrete(n = 4)
        self.observation_space = gym.spaces.Box(
            low = 0, high = 255, shape = sample.shape, dtype = np.uint8
        )
        self.thresholds = np.linspace(
            start = np.min(sample), stop = np.max(sample), num = n_thresholds, dtype = np.uint8
        )
        self.n_thresholds = n_thresholds
        self.tis = np.array([0, self.n_thresholds - 1])


    def _process_thresholds(self, action):
        return np.clip(self.tis + self.action_map[action], 0, self.n_thresholds - 1)


    def _compute_reward(self, predicted):
        height, width = self.label.shape

        return np.sum(np.logical_xor(predicted, self.label)) / (height * width)
    

    def reset(self, seed = None, options = None):
        super().reset(seed = seed)

        self.tis = np.array([0, self.n_thresholds - 1])
        self.dissim = self._compute_reward(np.full(self.sample.shape, 255))
        self.state = self.sample.copy()

        return self.sample.copy(), {}


    def step(self, action):
        self.tis = self._process_thresholds(action)
        lt, rt = self.thresholds[self.tis]

        predicted = cv.inRange(self.sample, int(lt), int(rt))
        self.state = cv.bitwise_and(self.sample, self.sample, mask = predicted)

        dissim = self._compute_reward(predicted)
        done = dissim < 0.05

        if dissim < self.dissim:
            reward = 10
        elif dissim == self.dissim:
            reward = 0
        elif dissim > self.dissim:
            reward = 0

        self.dissim = dissim

        return self.state, reward, done, False, {}
    
    
    def render(self):
        plt.imshow(self.state, cmap = 'gray', vmin = np.min(self.sample), vmax = np.max(self.sample))
        plt.show()