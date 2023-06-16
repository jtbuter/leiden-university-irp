import numpy as np

from irp.envs.env import Env

class MaskEnv(Env):
    def action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n)

        for action in range(self.action_space.n):
            ti = self.ti + self.action_mapping[action]

            # Check if this action would make the index go out of bounds
            mask[action] = ti >= 0 and ti < self.n_thresholds

        return mask
