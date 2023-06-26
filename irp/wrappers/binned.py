from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import gym
import gym.spaces
from gym.spaces import Space
import numpy as np
import irp.wrappers as wrappers

class Binned(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        bins_per_dim: Tuple[int, ...],
        limits: List[Tuple[float, ...]]
    ):
        super().__init__(env)

        self._observation_space = gym.spaces.MultiDiscrete(nvec=(1,) * len(bins_per_dim)) # TODO: Klopt deze?
        self._action_space = env.action_space
        self._grid = self.create_grid(limits, bins_per_dim)

        self.n_features = bins_per_dim

        self.env: gym.Env

    @classmethod
    def encode(
        self,
        grid: List[np.ndarray],
        state: Tuple[float, ...]
    ) -> np.ndarray:
        return np.asarray(list(int(np.digitize(s, g)) for s, g in zip(state, grid)))

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        state, reward, done, info = self.env.step(action)
        encoded = self.encode(self._grid, state)

        return encoded, reward, done, info

    def reset(self, **kwargs: Dict) -> np.ndarray:
        state, info = self.env.reset(**kwargs)
        encoded = self.encode(self._grid, state)

        return encoded, info

    @classmethod
    def create_grid(self, limits: List[Tuple[float, ...]], bins_per_dim: Tuple[int, ...]):
        limits = np.asarray(limits)
        lows = limits[:, 0]
        highs = limits[:, 1]

        return [np.linspace(lows[dim], highs[dim], bins_per_dim[dim] + 1)[1:-1] for dim in range(len(bins_per_dim))]
