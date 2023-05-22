from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import gym
import gym.spaces
from gym.spaces import Space
import numpy as np
import irp.wrappers as wrappers

if TYPE_CHECKING:
    from irp.wrappers.utils import IHT

class Tiled(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        tiles_per_dim: Tuple[int, ...],
        value_limits: List[Tuple[Union[float, int], ...]],
        tilings: int,
    ):
        super().__init__(env)

        self._observation_space = gym.spaces.MultiDiscrete(nvec=(1,) * tilings)

        self._action_space = env.action_space
        self._T = wrappers.utils.TileCoder(tiles_per_dim, value_limits, tilings)
        self._tilings = tilings

    @classmethod
    def encode(
        self,
        T: Union[IHT, int, None],
        state: List[float],
    ):
        return T[state]

    def step(self, action: int):
        state, reward, done, info = self.env.step(action)
        encoded = self.encode(self._T, state)

        return np.asarray(encoded), reward, done, info

    def reset(self, **kwargs: Dict):
        state = self.env.reset(**kwargs)
        encoded = self.encode(self._T, state)

        return np.asarray(encoded)

    @property
    def T(self):
        return self._T