import gym
import numpy as np
import cv2
import matplotlib.pyplot as plt

from irp import utils
from irp.envs.paper_ultra_sound_env import PaperUltraSoundEnv

from copy import deepcopy

class Paper2008UltraSoundEnv(PaperUltraSoundEnv):
    def __init__(self, sample = None, label = None, num_thresholds = 10, vjs = (0,)):
        super().__init__(sample, label, num_thresholds)

        self.vj = None
        self.vjs = np.array(vjs)
        self.action_map = self._vjs_step_map()
        self.action_space = gym.spaces.Discrete(n = len(self.action_map))

    def _vjs_step_map(self):
        action_map = []

        for vj in self.vjs:
            action_map.extend(((-1, vj), (1, vj)))

        return action_map

    def step(self, action):
        delta, vj = self.action_map[action]
        new_threshold_id = np.clip(self.threshold_ids + delta, 0, self.num_thresholds - 1)

        # Convert indices to gray-values for generalization
        ti = self.thresholds[new_threshold_id]

        # Extract a bit-mask using the gray-values
        bit_mask = cv2.threshold(self.sample, int(ti), 255, cv2.THRESH_BINARY_INV)[1]

        # Cast vj to an integer for cv2
        self.vj = int(vj)

        # Check that the structuring element has a size
        if self.vj != 0:
            # Apply an opening to the bit-mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.vj, self.vj))
            bit_mask = cv2.morphologyEx(bit_mask, cv2.MORPH_OPEN, kernel)

        # Convert the bit-mask to a discrete state
        next_state = self.observation(bit_mask)

        # Compute dissimilarity and convert this to a reward
        dissim = utils.compute_dissimilarity(bit_mask, self.label)
        reward = self._get_reward(dissim)
        is_done = bool(dissim < 0.05)

        self.old_dissim = dissim
        self.threshold_ids = new_threshold_id
        self.state = next_state

        return np.asarray(self.state, dtype=np.float32), reward, is_done, {}

    def reset(self):
        # Pick a new random threshold index
        new_threshold_id = np.random.choice(range(0, self.num_thresholds), 1)

        # Convert indices to gray-values for generalization
        ti = self.thresholds[new_threshold_id]

        # Extract a bit-mask using the gray-values
        bit_mask = cv2.threshold(self.sample, int(ti), 255, cv2.THRESH_BINARY_INV)[1]

        # Pick new morphing element size
        vj = np.random.choice(self.vjs, 1)

        # Cast vj to an int for cv2
        self.vj = int(vj)

        # Check that the structuring element has a size
        if self.vj != 0:
            # Apply an opening to the bit-mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.vj, self.vj))
            bit_mask = cv2.morphologyEx(bit_mask, cv2.MORPH_OPEN, kernel)

        # Convert the bit-mask to a discrete state
        next_state = self.observation(bit_mask)

        # Compute current dissimilarity
        dissim = utils.compute_dissimilarity(bit_mask, self.label)

        self.old_dissim = dissim
        self.threshold_ids = new_threshold_id
        self.state = next_state

        return np.asarray(self.state, dtype=np.float32)

    def render(self, mode = None):
        # Convert index to gray-values for generalization
        ti = self.thresholds[self.threshold_ids]

        # Extract a bit-mask using the gray-values
        state = cv2.threshold(self.sample, int(ti), 255, cv2.THRESH_BINARY_INV)[1]

        before_morph = deepcopy(state)

        # Check that the structuring element has a size
        if self.vj != 0:
            # Apply an opening to the bit-mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.vj, self.vj))
            state = cv2.morphologyEx(state, cv2.MORPH_OPEN, kernel)

        # Show the final result
        plt.imshow(np.hstack([self.label, before_morph, state]), cmap = 'gray', vmin = 0, vmax = 1)
        plt.show()