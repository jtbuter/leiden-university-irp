import numpy as np
from typing import Optional, Callable

from irp.policies.tiled_policy import TiledQ
from irp.envs.env import Env

class MaskTiledQ(TiledQ):
    def __init__(self, n_features: int, n_actions: int, alpha: float):
        super().__init__(n_features, n_actions, alpha)

    def predict(self, state: np.ndarray, mask_fn: Callable) -> int:
        values = self.values(state)
        mask = np.logical_not(mask_fn())

        values[mask] = -np.inf

        return self._argmax(values)
