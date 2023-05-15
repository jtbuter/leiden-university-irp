import numpy as np
import random
import math
import gym
from typing import List, Optional

class MultiSample(gym.Wrapper):
    def __init__(self, envs: Optional[List[gym.Env]] = []) -> None:
        # At least one environment was passed already, so we can initialize
        if len(envs) > 0:
            super().__init__(envs[0])
        
        self._envs = envs
        self._current_env = 0

    def reset(self, **kwargs):
        envs = self._envs

        self._env_id = math.floor(random.random() * len(envs))
        self.env = envs[self._env_id]

        return self.env.reset(**kwargs)

    def add(self, env: gym.Env) -> None:
        # This is the first environment that is added to the MultiSample list
        if len(self._envs) == 0:
            super().__init__(env)

        self._envs.append(env)

    @property
    def envs(self):
        return self._envs

    def __len__(self):
        return len(self._envs)

    def __getitem__(self, key):
        return self._envs[key]