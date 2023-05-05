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

        self._envs = envs

    def reset(self, **kwargs):
        # self._env_id = np.random.randint(len(self._envs))
        # self._env_id = math.floor(random.random() * len(self._envs))
        self._env_id = random.randrange(0, len(self._envs))
        self.env = self._envs[self._env_id]

        return super().reset(**kwargs)

    def add(self, env: gym.Env) -> None:
        # This is the first environment that is added to the MultiSample list
        if len(self._envs) == 0:
            super().__init__(env)

        self._envs.append(env)