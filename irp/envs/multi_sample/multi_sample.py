from typing import List

import matplotlib.pyplot as plt
import numpy as np
from gym import spaces

from irp import ROOT_DIR, GIT_DIR
from irp.envs import utils
from irp.envs.ultrasound.ultra_sound_env import UltraSoundEnv

class MultiSampleEnv(UltraSoundEnv):
    """
    Environment which can take multiple samples, and uses these to train
    and agent.
    """
    def __init__(
        self,
        samples: List[np.ndarray],
        labels: List[np.ndarray],
        n_thresholds: int
    ):
        self._samples = samples
        self._labels = labels
        self._n_thresholds = n_thresholds
        self._n_samples = len(self._samples)

        self._thresholds = None
        self.delta = 0.05

        self._build()

    def _build(self):
        self._action_map = np.array([(1,), (0,), (-1,)], dtype=np.int8)
        self.action_space = spaces.Discrete(n=self._action_map.size)

    def step(self, action: int):
        # Take a step to the left or right
        threshold_id_old = self._threshold_id
        self._threshold_id = utils.update_thresholds(
            action, self._action_map, threshold_id_old, self._n_thresholds
        )

        # Convert index to threshold-value and apply the threshold
        ti = int(self._thresholds[self._threshold_id])
        bit_mask = utils.apply_threshold(self._sample, ti)

        # Convert the bit-mask to a state-value
        self._state = self.observation(bit_mask)

        # Compute dissimilarity and convert this to a reward
        d_sim = utils.compute_dissimilarity(bit_mask, self._label)
        reward = self.reward(d_sim)
        self._d_sim = d_sim

        # Expose the dissimilarity for evaluation
        info = {'dissimilarity': self._d_sim}
        done = self._d_sim <= self.delta

        return np.asarray(self._state, dtype=np.float32), reward, done, info

    def _reward(self, d_sim):
        if self._d_sim <= self.delta:
            return 10
        else:
            return 0

    def _render(self):
        ti = self._thresholds[self._threshold_id]
        bit_mask = utils.apply_threshold(self._sample, ti)

        fig, (axl, axr) = plt.subplots(ncols=2)

        axl.imshow(bit_mask, cmap='gray', vmin=0, vmax=1)
        axr.imshow(self._label, cmap='gray', vmin=0, vmax=1)

        plt.show()

    def reset(self):
        # Index the subimages and labels randomly
        idx = np.random.randint(self._n_samples)
        self._label = self._labels[idx]
        self._sample = self._samples[idx]

        # Intensity cut-offs must be determined on reset, as the sample is picked here
        self._thresholds = utils.get_intensity_spectrum(self._sample, self._n_thresholds)

        # Always use the highest possible threshold value
        self._threshold_id = self._n_thresholds - 1

        # Apply the threshold
        bit_mask = utils.apply_threshold(self._sample, ti_left=self._threshold_id)

        # Convert the bit-mask to a state-value
        self._state = self.observation(bit_mask)

        # Compute the starting dissimilarity
        self._d_sim = utils.compute_dissimilarity(bit_mask, self._label)

        return np.asarray(self._state, dtype=np.float32)
