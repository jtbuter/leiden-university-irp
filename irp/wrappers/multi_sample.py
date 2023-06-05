from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import random, math
import gym

class MultiSample(gym.Wrapper):
    def __init__(self, envs: List[gym.Env] = []):
        # No environments were actually passed yet
        if len(envs) > 0:
            super().__init__(envs[0])
        
        self._envs = envs

    def reset(self, **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        self._env_id = math.floor(random.random() * len(self._envs))

        self.env = self._envs[self._env_id]

        return self.env.reset(**kwargs)

    def add(self, env: gym.Env):
        # This is the first environment that is added to the MultiSample list
        if len(self._envs) == 0:
            super().__init__(env)

        self._envs.append(env)