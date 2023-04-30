import numpy as np
import gym
from typing import List

class MultiSample(gym.Wrapper):
    def __init__(self, envs: List[gym.Env]) -> None:
        super().__init__(envs[0])

        self._envs = envs

    def reset(self, **kwargs):
        self._env_id = np.random.randint(len(self._envs))
        self.env = self._envs[self._env_id]

        return super().reset(**kwargs)