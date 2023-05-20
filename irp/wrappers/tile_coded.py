from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import gym
import gym.spaces
from gym.spaces import Space
import numpy as np
import irp.wrappers as wrappers

class Tiled(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        tiles_per_dim: Tuple[int, ...],
        tilings: int,
        limits: List[Tuple[float, ...]]
    ):
        super().__init__(env)

        self._T = wrappers.utils.TileCoder(tiles_per_dim, limits, tilings)
        self._observation_space = gym.spaces.MultiDiscrete(nvec=(1,) * tilings)

        self._action_space = env.action_space
        self._tilings = tilings

        self.env: gym.Env

    @property
    def T(self) -> wrappers.utils.TileCoder:
        return self._T

    @classmethod
    def encode(
        self,
        T: wrappers.utils.TileCoder,
        state: Tuple[float, ...]
    ) -> np.ndarray:
        return T[state]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        state, reward, done, info = self.env.step(action)
        encoded = self.encode(self._T, state)

        return encoded, reward, done, info

    def reset(self, **kwargs: Dict) -> np.ndarray:
        state = self.env.reset(**kwargs)
        encoded = self.encode(self._T, state)

        return encoded