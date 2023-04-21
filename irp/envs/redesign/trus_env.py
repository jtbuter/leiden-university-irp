import matplotlib.pyplot as plt
import numpy as np
import cv2

import gym
import gym.wrappers

from irp.envs.ultrasound.ultra_sound_env import UltraSoundEnv
from irp.wrappers import Discretize
from irp import ROOT_DIR, utils

class TrusEnv(UltraSoundEnv):
    # Action map which allows for modifying the threshold index
    action_map = np.array([
        (-1, 1), (1, -1),
        (-1, -1), (1, 1),
        (0, 0) # Neutral action that doesn't modify
    ])

    def __init__(
        self,
        sample: np.ndarray,
        label: np.ndarray,
        num_thresholds: int,
        max_unimproved_steps: int = np.inf
    ):
        super().__init__(sample, label, num_thresholds)

        self.action_space = gym.spaces.Discrete(n=len(self.action_map))

        # Counts number of times we didn't improve the accuracy
        self.num_unimproved_steps = 0
        self.max_unimproved_steps = max_unimproved_steps

    def step(self, action: int):
        # Convert an action to new threshold indices
        new_threshold_ids = utils.process_thresholds(
            action, self.action_map, self.threshold_ids, self.num_thresholds
        )

        # # If the action we're trying to perform is not valid; do nothing
        # if not self._is_valid_action(*new_threshold_ids):
        #     new_threshold_ids = self.threshold_ids

        # Convert indices to gray-values
        lt, rt = self.thresholds[new_threshold_ids]

        # Extract a bit-mask using the gray-values
        bit_mask = utils.apply_threshold(self.sample, lt, rt)

        # Convert the bit-mask to a discrete state
        next_state = self.observation(bit_mask)

        # Compute dissimilarity and convert this to a reward
        dissim = utils.compute_dissimilarity(bit_mask, self.label)

        # We made an improvement
        if dissim < self.old_dissim:
            self.num_unimproved_steps = 0
        else:
            self.num_unimproved_steps += 1

        done = self._is_done(dissim)
        reward = self.reward(dissim)

        self.old_dissim = dissim
        self.threshold_ids = new_threshold_ids
        self.state = next_state

        info = {'dissimilarity': dissim}

        return np.asarray(next_state, dtype=np.float32), reward, done, info

    # def observation(self, bit_mask):
    #     return super().observation(bit_mask)[:2]

    def _is_done(self, dissim):
        if self.num_unimproved_steps >= self.max_unimproved_steps:
            return True

        return bool(dissim < 0.05)

    def _reward(self, dissim):
        if self.num_unimproved_steps >= self.max_unimproved_steps:
            return -1

        if dissim <= self.old_dissim:
            return 1
        elif dissim > self.old_dissim:
            return -1

    def reset(self):
        # Pick two random new threshold indices
        new_threshold_ids = np.random.choice(range(0, self.num_thresholds), 2)

        # Ensuring the left threshold is always smaller than the right threshold
        new_threshold_ids = np.sort(new_threshold_ids)

        # Convert indices to gray-values for generalization
        lt, rt = self.thresholds[new_threshold_ids]

        # Extract a bit-mask using the gray-values
        bit_mask = utils.apply_threshold(self.sample, lt, rt)

        # Convert the bit-mask to a discrete state
        next_state = self.observation(bit_mask)

        # Compute current dissimilarity
        dissim = utils.compute_dissimilarity(bit_mask, self.label)

        self.num_unimproved_steps = 0
        self.old_dissim = dissim
        self.threshold_ids = new_threshold_ids
        self.state = next_state

        return np.asarray(self.state, dtype=np.float32)

    def _render(self):
        lt, rt = self.thresholds[self.threshold_ids]

        # Extract a bit-mask using the gray-values
        state = utils.apply_threshold(self.sample, lt, rt)

        # Create a canvas to draw to
        fig, (label_axis, predict_axis) = plt.subplots(1, 2)

        # Show the images
        label_axis.imshow(self.label, cmap='gray', vmin=0, vmax=1)
        predict_axis.imshow(state, cmap='gray', vmin=0, vmax=1)

        # Set titles
        label_axis.title.set_text('Label')
        predict_axis.title.set_text('Prediction')

        plt.show()