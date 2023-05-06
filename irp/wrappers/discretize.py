from typing import Dict, List, Tuple, Union
import gym
from gym.spaces import Space
import numpy as np
import irp.wrappers as wrappers

class Discretize(gym.Wrapper):
    def __init__(self, env: gym.Env, lows: Union[Dict, List], highs: Union[Dict, List], bins: Tuple):
        super().__init__(env)

        self._observation_space = gym.spaces.MultiDiscrete(bins)
        self._action_space = env.action_space

        self._state_bins = None

        self._setup_env(lows, highs)

    def _setup_env(self, lows: Union[Dict, List], highs: Union[Dict, List]):
        self._state_bins = self.make_state_bins(self.observation_space, lows, highs)

    @classmethod
    def make_state_bins(
        self,
        dims: Union[Space, Tuple],
        lows: Union[Dict, List],
        highs: Union[Dict, List]
    ):
        if isinstance(lows, dict):
            keys = lows.keys()
        else:
            keys = range(len(lows))

        if isinstance(dims, Space):
            dims = wrappers.utils.get_dims(dims)

        bins = dict(zip(keys, dims))

        return [np.linspace(lows[key], highs[key], bins[key] + 1)[1:-1] for key in keys]

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        return np.asarray(wrappers.utils.discrete(state, self._state_bins)), reward, done, info

    def reset(self, **kwargs):
        state = self.env.reset(**kwargs)

        return np.asarray(wrappers.utils.discrete(state, self._state_bins))