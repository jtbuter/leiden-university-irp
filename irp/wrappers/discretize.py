from typing import Dict, List, Tuple, Union
import gym
import numpy as np
import irp.utils

class Discretize(gym.Wrapper):
    def __init__(self, env: gym.Env, lows: Union[Dict, List], highs: Union[Dict, List], bins: Tuple):
        super().__init__(env)

        self._observation_space = gym.spaces.MultiDiscrete(bins)
        self._action_space = env.action_space

        self._state_bins = None

        self._setup_env(lows, highs)

    def _setup_env(self, lows: Union[Dict, List], highs: Union[Dict, List]):
        if isinstance(lows, dict):
            keys = lows.keys()
        else:
            keys = range(len(lows))

        bins = dict(zip(keys, irp.utils.get_dims(self.observation_space)))

        self._state_bins = [
            np.linspace(lows[key], highs[key], bins[key] + 1)[1:-1] for key in keys
        ]

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        return np.asarray(irp.utils.discrete(state, self._state_bins)), reward, done, info

    def reset(self):
        state = self.env.reset()

        return np.asarray(irp.utils.discrete(state, self._state_bins))