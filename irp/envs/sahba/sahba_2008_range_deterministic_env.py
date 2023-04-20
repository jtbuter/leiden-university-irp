from typing import Tuple
import gym
import numpy as np
import cv2
import matplotlib.pyplot as plt

import irp.utils
from irp.envs.sahba.sahba_2008_range_threshold_env import Sahba2008RangedEnv

class Sahba2008RangedDeterministicEnv(Sahba2008RangedEnv):
    def reset(self):
        # Pick a random intensity
        ti = 0

        # Pick new morphing element size
        vj = 0

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
