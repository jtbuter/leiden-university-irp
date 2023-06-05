from typing import Callable, Dict, Optional
import numpy as np
import gym

import irp.policies.tiled_policy

class Qlearning():
    def __init__(self, environment: gym.Env):
        self.environment = environment
        self.t = 0
        self.e = 0

    def learn(
        self,
        max_t: int, max_e: int, alpha: float,
        ep_max: float = 1.0, ep_min: float = 0.1,
        ep_frac: float = 0.0, gamma: float = 0.95, 
        callback: Optional[Dict[str, any]] = None
    ):
        self.policy = irp.policies.tiled_policy.TiledQ(self.environment.T.n_tiles, self.environment.action_space.n, alpha=alpha)
        t, e = 0, 0
        ep = ep_max
        continue_training = True

        for e in range(max_e):
            state, info = self.environment.reset()
            done = False

            while not done:
                if np.random.random() < ep:
                    action = self.environment.action_space.sample()
                else:
                    action = self.policy.predict(state)

                next_state, reward, done, info = self.environment.step(action)
                target = reward + gamma * max(self.policy.values(next_state)) * (not done)

                self.policy.update(state, action, target)

                state = next_state

                self.t += 1

            self.e = e + 1

            ep = max(ep_min, ep - ep_frac)

            if callback is not None and e % callback['interval'] == 0:
                continue_training = callback['callback'](locals())
            
            if continue_training is False:
                break

        return self.policy