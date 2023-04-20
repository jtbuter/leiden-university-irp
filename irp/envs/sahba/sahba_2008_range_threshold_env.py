from typing import Tuple
import gym
import numpy as np
import cv2
import matplotlib.pyplot as plt

import irp.utils
from irp.envs.sahba.sahba_2008_env import Sahba2008UltraSoundEnv

class Sahba2008RangedEnv(Sahba2008UltraSoundEnv):
    def __init__(
        self,
        sample: np.ndarray = None,
        label: np.ndarray = None,
        num_thresholds: int = None,
        vjs: Tuple = None
    ):
        super().__init__(sample, label, num_thresholds, vjs)

        # Store the used intensity threshold
        self.ti = None

        # Create the action map
        self.action_map = self._ranged_step_map()

        # Update the action-space based on the new action map
        self.action_space = gym.spaces.Discrete(n=len(self.action_map))

    def step(self, action):
        ti, vj = self.action_map[action]

        # Cast vj and ti to an integer for cv2
        self.vj = int(vj)
        self.ti = int(ti)

        # Apply the threshold and obtain bit mask
        bit_mask = cv2.threshold(self.sample, self.ti, 255, cv2.THRESH_BINARY_INV)[1]

        # Apply a morphological opening
        bit_mask = self._apply_opening(bit_mask, self.vj)

        # Convert the bit-mask to a discrete state
        next_state = self.observation(bit_mask)

        # Compute dissimilarity and convert this to a reward
        dissim = irp.utils.compute_dissimilarity(bit_mask, self.label)
        reward = self.reward(dissim)
        is_done = bool(dissim < 0.05)

        # Save values for the potential subsequent step
        self.old_dissim = dissim
        self.state = next_state

        return np.asarray(self.state, dtype=np.float32), reward, is_done, {}

    def reset(self):
        # Pick a random intensity
        ti = np.random.choice(self.thresholds, 1)

        # Pick new morphing element size
        vj = np.random.choice(self.vjs, 1)

        # Cast vj and ti to an integer for cv2
        self.vj = int(vj)
        self.ti = int(ti)

        # Extract a bit-mask using the gray-values
        bit_mask = cv2.threshold(self.sample, self.ti, 255, cv2.THRESH_BINARY_INV)[1]

        # Apply a morphological opening
        bit_mask = self._apply_opening(bit_mask, self.vj)

        # Convert the bit-mask to a discrete state
        next_state = self.observation(bit_mask)

        # Compute current dissimilarity
        dissim = irp.utils.compute_dissimilarity(bit_mask, self.label)

        self.old_dissim = dissim
        self.state = next_state

        return np.asarray(self.state, dtype=np.float32)

    def _render(self):
        # Extract a bit-mask using the gray-values
        state = cv2.threshold(self.sample, self.ti, 255, cv2.THRESH_BINARY_INV)[1]

        # Apply a morphological opening
        state = self._apply_opening(state, self.vj)

        # Create a canvas to draw to
        fig, (label_axis, predict_axis) = plt.subplots(1, 2)

        # Show the images
        label_axis.imshow(self.label, cmap='gray', vmin=0, vmax=1)
        predict_axis.imshow(state, cmap='gray', vmin=0, vmax=1)

        # Set titles
        label_axis.title.set_text('Label')
        predict_axis.title.set_text('Prediction')

        plt.show()

    def _ranged_step_map(self):
        action_map = [(ti, vj) for vj in self.vjs for ti in self.thresholds]

        return action_map

