import gym
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

class UltraSoundEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    action_map = np.array([(-1, -1), (-1, 1), (1, -1), (1, 1)])

    def __init__(self, sample = None, label = None, n_thresholds = 10):
        super(UltraSoundEnv, self).__init__()

        self.state = sample.copy()
        self.label = label.copy()
        self.sample = sample.copy()

        self.action_space = gym.spaces.Discrete(n = 4)
        self.observation_space = gym.spaces.Box(
            low = 0, high = 255, shape = sample.shape + (1,), dtype = np.uint8
        )

        self.n_thresholds = n_thresholds
        self.thresholds = np.linspace(
            np.min(sample), np.max(sample), n_thresholds, dtype = np.uint8
        ) # TODO: mogelijk in-place berekening van thresholds, i.p.v. deze array gebruiken
        self.tis = np.array([0, self.n_thresholds - 1])

        self.dissim = self._compute_reward(np.full(label.shape, 255))


    def _process_thresholds(self, action):
        return np.clip(self.tis + self.action_map[action], 0, self.n_thresholds - 1)


    def _compute_reward(self, bit_mask):
        height, width = self.label.shape

        return np.sum(np.logical_xor(bit_mask, self.label)) / (height * width)


    def step(self, action):
        tis = self._process_thresholds(action)
        lt, rt = self.thresholds[tis]

        bit_mask = cv.inRange(self.sample, int(lt), int(rt))
        next_state = cv.bitwise_and(self.sample, self.sample, mask = bit_mask)

        dissim = self._compute_reward(bit_mask)
        is_done = bool(dissim < 0.05)

        if dissim < self.dissim:
            reward = 10
        elif dissim == self.dissim:
            reward = 0
        elif dissim > self.dissim:
            reward = 0

        self.dissim = dissim
        self.state = next_state.reshape(*next_state.shape, 1)
        self.tis = tis

        return self.state, reward, is_done, {}


    def reset(self):
        self.dissim = self._compute_reward(np.full(self.label.shape, 255))
        self.state = self.sample.reshape(*self.sample.shape, 1)
        self.tis = np.array([0, self.n_thresholds - 1])

        return self.state


    def render(self, mode = 'human'):
        plt.imshow(np.squeeze(self.state), cmap = 'gray', vmin = 0, vmax = 1)
        plt.show()


    def close (self):
        pass