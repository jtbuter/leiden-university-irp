from typing import List

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
        self.samples = samples
        self.labels = labels
        self.n_thresholds = n_thresholds

        self.thresholds = None

        self._build()

    def _build(self):
        # Cast the floating point thresholds from the base-class to integers
        self.action_map = np.array([-1, 0, 1])
        self.action_space = spaces.Discrete(n=self.action_map.size)

    def step(self):
        pass

    def reset(self):
        # Pick a random sample
        sample = np.random.choice(self.samples, size=1)

        # Intensity cut-offs must be determined on reset, as the sample is picked here
        self.thresholds = utils.get_intensity_spectrum(sample, self.n_thresholds)

        # Always use the highest possible threshold value
        self.threshold_id = self.n_thresholds - 1

        # Apply the threshold
        bit_mask = utils.apply_threshold(sample, ti_left=self.threshold_id)

        # Convert the bit-mask to a state-value


        return


