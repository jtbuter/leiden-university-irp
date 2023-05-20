from __future__ import annotations

from typing import List, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np

import gym

import irp.wrappers as wrappers

class TiledQTable():
    def __init__(self, environment: gym.Env):
        self.qtable = self._build_qtable(environment)

        self.environment = environment
        self.action_space = environment.action_space

    def _build_qtable(self, environment: gym.Env) -> np.ndarray:
        dims = (environment.T.n_tiles,) + wrappers.utils.get_dims(environment.action_space)

        print(dims)

        return np.zeros(dims)

    def state_values(self, state: List[int]) -> List[float]:
        return [self.value(state, a) for a in range(self.action_space.n)]

    def value(self, state: List[int], action: int) -> float:
        return self.qtable[state, action].mean()
        
        # # The terminal state should be absorbing; resolving to True means the V(s) value remains 0.0
        # if self.environment.is_terminal(state):
        #     value = 0.0
        # else:
        #     value = self.qtable[state, action].mean()

        # return value

    def update(self, state: List[int], action: int, target: float, alpha: float):
        import warnings
        warnings.filterwarnings('error')

        # estimate = 0.0

        # for tile in state:
        #     estimate += self.qtable[tile, action]

        # error = target - estimate

        # for tile in state:
        #     self.qtable[tile, action] += alpha * error

        for tile in state: # TODO Deze manier weer gebruiken
            try:
                self.qtable[tile, action] += alpha * (target - self.qtable[tile, action])
            except Warning:
                print(self.qtable[tile, action], alpha * (target - self.qtable[tile, action]))

                raise Exception()