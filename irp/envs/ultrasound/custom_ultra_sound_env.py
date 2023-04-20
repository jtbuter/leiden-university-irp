import numpy as np
import cv2

import irp.utils
from irp.envs.ultrasound.ultra_sound_env import UltraSoundEnv

class CustomUltraSoundEnv(UltraSoundEnv):
    def step(self, action):
        # Convert an action to new threshold indices
        new_threshold_ids = irp.utils.process_thresholds(action, self.action_map, self.threshold_ids, self.num_thresholds)

        # If the action we're trying to perform is not valid; do nothing
        if not self._is_valid_action(*new_threshold_ids):
            new_threshold_ids = self.threshold_ids

        # Convert indices to gray-values for generalization
        lt, rt = self.thresholds[new_threshold_ids]

        # Extract a bit-mask using the gray-values
        bit_mask = cv2.inRange(self.sample, int(lt), int(rt))

        # Convert the bit-mask to a discrete state
        next_state = self.observation(bit_mask)

        # Compute dissimilarity and convert this to a reward
        dissim = irp.utils.compute_dissimilarity(bit_mask, self.label)
        reward = self.reward(dissim)
        is_done = bool(dissim < 0.05)

        self.old_dissim = dissim
        self.threshold_ids = new_threshold_ids
        self.state = next_state

        return np.asarray(self.state, dtype=np.float32), reward, is_done, {}

    def reset(self):
        # Pick two random new threshold indices
        new_threshold_ids = np.random.choice(range(0, self.num_thresholds), 2)

        # Ensuring the left threshold is always smaller than the right threshold
        new_threshold_ids = np.sort(new_threshold_ids)

        # Convert indices to gray-values for generalization
        lt, rt = self.thresholds[new_threshold_ids]

        # Extract a bit-mask using the gray-values
        bit_mask = cv2.inRange(self.sample, int(lt), int(rt))

        # Convert the bit-mask to a discrete state
        next_state = self.observation(bit_mask)

        # Compute current dissimilarity
        dissim = irp.utils.compute_dissimilarity(bit_mask, self.label)

        self.old_dissim = dissim
        self.threshold_ids = new_threshold_ids
        self.state = next_state

        return np.asarray(self.state, dtype=np.float32)