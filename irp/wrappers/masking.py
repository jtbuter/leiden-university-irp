from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import gym
import gym.spaces
from gym.spaces import Space
import numpy as np
import irp.wrappers as wrappers

class MaskedDiscrete(gym.spaces.Discrete):
    def __init__(
        self,
        n: int,
        mask_fn: Callable
    ) -> None:
        super().__init__(n)

        self.mask_fn = mask_fn

    def sample(self) -> int:
        action = self._sample()

        # Check if any action was allowed, otherwise sample randomly
        if action is not None:
            return action
        else:
            return super().sample()

    def _sample(self) -> Union[int, None]:
        mask = self.mask_fn()
        valid_action_mask = mask == 1

        if np.any(valid_action_mask):
            return int(self.np_random.choice(np.where(valid_action_mask)[0]))

        return None

class ActionMasker(gym.Wrapper):
    def __init__(self, env: gym.Env, mask_fn: Optional[Callable] = None) -> None:
        super().__init__(env)

        if mask_fn is None:
            # Out of the allowed actions, select the ones that give a positive reward
            mask_fn = lambda env: env.action_mask()[env.guidance_mask()]

        self._action_space = MaskedDiscrete(env.action_space.n, lambda: mask_fn(env))

    @property
    def action_space(self) -> MaskedDiscrete:
        return self._action_space