from typing import Callable, Optional, Tuple
import numpy as np

class Q():
    def __init__(self, n_features: Tuple[int, ...], n_actions: int, alpha: float):
        self.w = np.zeros(n_features + (n_actions,))

        self._n_features = n_features
        self._n_actions = n_actions
        self._alpha = alpha

        self.w: np.ndarray

    def predict(self, state: np.ndarray, mask_fn: Callable, deterministic: Optional[bool] = False) -> int:
        values = self.values(state)
        mask = np.logical_not(mask_fn())

        values[mask] = -np.inf

        # Allow for tie-breacking through randomness
        if deterministic is False:
            return self._argmax(values)
        else:
            return np.argmax(values)

    def values(self, state: np.ndarray) -> np.ndarray:
        values = [self.value(state, action) for action in range(self._n_actions)]

        return np.asarray(values)

    def value(self, state: np.ndarray, action: int) -> float:
        return self.w[tuple(state)][action]

    def update(self, state: np.ndarray, action: int, target: float):
        value = self.value(state, action)
        delta = self._alpha * (target - value)

        self.w[tuple(state)][action] += delta

    def _argmax(self, values: np.ndarray) -> int:
        # Break ties, np.argmax() always defaults to picking index 0 for ties, this picks randomly
        return np.random.choice(np.flatnonzero(values == np.max(values)))