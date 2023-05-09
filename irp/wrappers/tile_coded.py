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
        lows: Union[Dict, List],
        highs: Union[Dict, List],
        tilings: int,
        iht: Optional[Union[IHT, int]] = None,
        rescale: Optional[bool] = False
    ):
        super().__init__(env)

        if iht is None:
            self._observation_space = wrappers.utils.get_dims(env.observation_space) + (1,)
        elif isinstance(iht, wrappers.utils.IHT):
            self._observation_space = gym.spaces.Discrete(n=iht.size)
        elif isinstance(iht, int):
            self._observation_space = gym.spaces.Discrete(n=iht)

        self._action_space = env.action_space
        self._IHT = iht
        self._tilings = tilings
        self._rescale = rescale

        self._setup_env(lows, highs)

    def _setup_env(self, lows: Union[Dict, List], highs: Union[Dict, List]):
        self._input_limits = np.array([lows, highs])

    @classmethod
    def encode(
        self,
        iht: Union[IHT, int, None],
        tilings: int,
        state: List[float],
    ):
        return wrappers.utils.tiles(iht, tilings, state)

    def step(self, action: int):
        state, reward, done, info = self.env.step(action)

        if self._rescale:
            scaled_state = wrappers.utils.min_max_scaling(state, *self._input_limits)

        encoded = self.encode(self._IHT, self._tilings, scaled_state)

        return np.asarray(encoded), reward, done, info

    def reset(self, **kwargs: Dict):
        state = self.env.reset(**kwargs)

        if self._rescale:
            scaled_state = wrappers.utils.min_max_scaling(state, *self._input_limits)

        encoded = self.encode(self._IHT, self._tilings, scaled_state)

        return np.asarray(encoded)