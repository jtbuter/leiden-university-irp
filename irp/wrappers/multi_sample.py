import numpy as np
import random
import math
import gym
from typing import List, Optional

class MultiSample(gym.Wrapper):
    def __init__(self, envs: Optional[List[gym.Env]] = []) -> None:
        # No environments were actually passed yet
        if len(envs) > 0:
            super().__init__(envs[0])
        
        self._order = None
        self._envs = envs
        self._env_id = 0

    def reset(self, **kwargs):
        envs = self._envs

        if self._order is None:
            self._order = np.random.choice(range(len(envs)), len(envs), replace=False)

        # self._env_id = np.random.randint(len(envs))
        # self._env_id = math.floor(random.random() * len(envs))
        self._env_id = (self._env_id + 1) % len(envs)
        self.env = envs[self._order[self._env_id]]

        return super().reset(**kwargs)

    def add(self, env: gym.Env) -> None:
        # This is the first environment that is added to the MultiSample list
        if len(self._envs) == 0:
            super().__init__(env)

        self._envs.append(env)