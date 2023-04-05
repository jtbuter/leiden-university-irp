import gym
from stable_baselines3.common.vec_env import util
import numpy as np
import irp.utils

class ExpandDimsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = gym.spaces.MultiDiscrete((env.observation_space.n,))

    def step(self, action):
        obs, reward, terminated, info = self.env.step(action)

        return np.array([obs]), reward, terminated, info

    def reset(self):
        return np.array([self.env.reset()])

class Discretize(gym.Wrapper):
    def __init__(self, env, lows, highs, bins):
        super().__init__(env)

        self._observation_space = gym.spaces.MultiDiscrete(bins)
        self._action_space = env.action_space

        self._state_bins = None

        self._setup_env(lows, highs)

    def _setup_env(self, lows, highs):
        keys = lows.keys()
        bins = dict(zip(keys, irp.utils.get_dims(self.observation_space)))

        self._state_bins = np.asarray([
            np.linspace(lows[key], highs[key], bins[key] + 1)[1:-1] for key in keys
        ])

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        return np.asarray(irp.utils.discretize(state, self._state_bins)), reward, done, info

    def reset(self):
        state = self.env.reset()

        return np.asarray(irp.utils.discretize(state, self._state_bins))
