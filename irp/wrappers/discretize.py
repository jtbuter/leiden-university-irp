import gym
import numpy as np
from irp import utils

class Discretize(gym.Wrapper):
    def __init__(self, env, lows, highs, bins):
        super().__init__(env)

        self._observation_space = gym.spaces.MultiDiscrete(bins)
        self._action_space = env.action_space

        self._state_bins = None

        self._setup_env(lows, highs)

    def _setup_env(self, lows, highs):
        keys = lows.keys()
        bins = dict(zip(keys, utils.get_dims(self.observation_space)))

        self._state_bins = np.asarray([
            np.linspace(lows[key], highs[key], bins[key] + 1)[1:-1] for key in keys
        ])

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        return np.asarray(utils.discretize(state, self._state_bins)), reward, done, info

    def reset(self):
        state = self.env.reset()

        return np.asarray(utils.discretize(state, self._state_bins))
