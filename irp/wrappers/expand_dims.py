import gym
import numpy as np

class ExpandDims(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = gym.spaces.MultiDiscrete((env.observation_space.n,))

    def step(self, action):
        obs, reward, terminated, info = self.env.step(action)

        return np.array([obs]), reward, terminated, info

    def reset(self, **kwargs):
        state = self.env.reset(**kwargs)

        return np.array([state])
