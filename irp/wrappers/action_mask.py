from typing import Callable, Optional, Union

import numpy as np

import gym
import gym.spaces

class MaskedDiscrete(gym.spaces.Discrete):
    def __init__(
        self,
        n: int,
        mask_fn: Callable
    ) -> None:
        super().__init__(n)

        self.mask_fn = mask_fn

    def sample(self) -> int:
        mask = self.mask_fn()
        valid_action_mask = mask == 1

        if np.any(valid_action_mask):
            return int(self.np_random.choice(np.where(valid_action_mask)[0]))
        else:
            return 0

class ActionMasker(gym.Wrapper):
    def __init__(self, env: gym.Env, mask_fn: Callable) -> None:
        super().__init__(env)

        self._action_space = MaskedDiscrete(env.action_space.n, lambda: mask_fn(env))

    @property
    def action_space(self) -> MaskedDiscrete:
        """Returns the action space of the environment."""
        if self._action_space is None:
            return self.env.action_space

        return self._action_space