from typing import Callable, Dict, Optional, Union
import numpy as np
import gym

from irp.envs.env import Env
from irp.wrappers.masking import ActionMasker
from irp.wrappers.tiled import Tiled
from irp.policies.tiled_policy import TiledQ
from irp.policies.mask_tiled_policy import MaskTiledQ

class Sarsa():
    def __init__(self, environment: Union[gym.Env, Env, ActionMasker, Tiled]):
        self.environment = environment
        self.t = 0
        self.e = 0

    def learn(
        self,
        max_t: int, max_e: int, alpha: float,
        eps_max: float = 1.0, eps_min: float = 0.1,
        eps_frac: float = 0.0, gamma: float = 0.95, 
        callback: Optional[Dict[str, any]] = None,
        policy_cls = MaskTiledQ
    ):
        self.policy = policy_cls(self.environment.n_features, self.environment.n_tiles, self.environment.action_space.n, alpha=alpha)

        eps = eps_max
        continue_training, time_exceeded = True, False

        while self.e < max_e:
            state, info = self.environment.reset()
            done = False

            if np.random.random() < eps:
                action = self.environment.action_space.sample()
            else:
                action = self.policy.predict(state, self.environment.action_mask)

            while not done:
                next_state, reward, done, info = self.environment.step(action)

                if np.random.random() < eps:
                    next_action = self.environment.action_space.sample()
                else:
                    next_action = self.policy.predict(next_state, self.environment.action_mask)

                target = reward + gamma * self.policy.value(next_state, next_action)

                self.policy.update(state, action, target)

                state, action = next_state, next_action

                self.t += 1
                
                eps = max(eps_min, eps * eps_frac)

                time_exceeded = self.t >= max_t

                if time_exceeded:
                    break

            self.e += 1

            if callback is not None and self.e % callback['interval'] == 0:
                continue_training = callback['callback'](locals())
            
            if continue_training is False or time_exceeded:
                break

        return self.policy