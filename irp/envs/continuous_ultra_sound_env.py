import gym
import numpy as np
import cv2

from irp.envs.ultra_sound_env import UltraSoundEnv
from irp import utils

class ContinuousUltraSoundEnv(UltraSoundEnv):
    metadata = {
        "render.modes": ['human'], "render_fps": 1
    }
    action_map = np.array([
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 0), (0, 1),
        (1, -1), (1, 0), (1, 1)
    ])

    def __init__(self, sample = None, label = None, num_thresholds = 10, render_mode=None):
        super().__init__(sample, label, num_thresholds, render_mode)

        # Change the observation space to match images
        self.observation_space = gym.spaces.Box(0, 255, self.sample.shape + (1,), np.uint8)

    def _reward(self, dissim):
        super()._reward(dissim)

    def step(self, action):
        new_threshold_ids = utils.process_thresholds(action, self.action_map, self.threshold_ids, self.num_thresholds)
        lt, rt = self.thresholds[new_threshold_ids]

        bit_mask = cv2.inRange(self.sample, int(lt), int(rt))
        next_state = cv2.bitwise_and(self.sample, self.sample, mask = bit_mask)

        dissim = utils.compute_dissimilarity(bit_mask, self.label)
        reward = self.reward(dissim)
        is_done = bool(dissim < 0.05)

        self.old_dissim = dissim
        self.threshold_ids = new_threshold_ids
        self.state = next_state.reshape(*next_state.shape, 1)

        return self.state, reward, is_done, {}

    def reset(self):
        bit_mask = np.full(self.label.shape, 255)
        self.state = self.sample.reshape(*self.sample.shape, 1)
        self.threshold_ids = np.array([0, self.num_thresholds - 1])
        self.old_dissim = utils.compute_dissimilarity(bit_mask, self.label)

        return self.state

    def close(self):
        pass
