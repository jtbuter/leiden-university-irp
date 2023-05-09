from __future__ import annotations

from typing import List, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np

import gym

import irp.experiments.tile_coding.env as env
import irp.wrappers as wrappers
import irp.experiments.tile_coding.q as q

if TYPE_CHECKING:
    from irp.wrappers.utils import IHT

class TiledQTable():
    def __init__(self, environment: gym.Env, tilings: int, iht: Union[IHT, int, None]):
        self.tilings = tilings
        self.qtable = self._build_qtable(environment, iht)

        self.action_space = environment.action_space

    def _build_qtable(self, environment: gym.Env, iht: Union[IHT, int, None]) -> np.ndarray:
        if iht is None:
            dims = wrappers.utils.get_dims(environment.observation_space, environment.action_space)
        elif isinstance(iht, int):
            dims = (iht,) + wrappers.utils.get_dims(environment.action_space)
        elif isinstance(iht, IHT):
            dims = (iht.size,) + wrappers.utils.get_dims(environment.action_space)

        return np.zeros(dims)

    def qs(self, state: List[int]) -> int:
        return [self.value(state, a) for a in range(self.action_space.n)]

    def value(self, state: List[int], action: int) -> float:
        value = 0.0

        for tile in state:
                value += self.qtable[tile, action]

        return value / self.tilings

    def update(self, state: List[int], action: int, target: float, alpha: float):
        # estimate = 0.0

        # for tile in state:
        #     estimate += self.qtable[tile, action]

        # error = target - estimate

        # for tile in state:
        #     self.qtable[tile, action] += alpha * error

        for tile in state: # TODO Deze manier weer gebruiken
            self.qtable[tile, action] += alpha * (target - self.qtable[tile, action])