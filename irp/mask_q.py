from typing import Optional, TypeVar, Union

from stable_baselines3.common.type_aliases import GymEnv, Schedule

import numpy as np

from irp.q import Q
from irp.policies import MaskedQPolicy
from irp.wrappers import ActionMasker

SelfMaskedQ = TypeVar("SelfMaskedQ", bound="MaskedQ")

class MaskedQ(Q):
    def __init__(
        self,
        env: ActionMasker,
        learning_rate: Union[float, Schedule],
        gamma: float = 0.99,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        monitor_wrapper: bool = True,
        seed: Optional[int] = None,
        use_sb3_env = True,
        init_setup_model=True
    ):
        super().__init__(env, learning_rate, gamma=gamma, exploration_fraction=exploration_fraction, exploration_initial_eps=exploration_initial_eps, exploration_final_eps=exploration_final_eps, tensorboard_log=tensorboard_log, verbose=verbose, monitor_wrapper=monitor_wrapper, seed=seed, use_sb3_env=use_sb3_env, init_setup_model=init_setup_model)

        self.policy = MaskedQPolicy(env.observation_space, env.action_space)
        self._mask_fn = env.action_space.mask_fn

        self.policy: MaskedQPolicy

    def _setup_model(self: SelfMaskedQ) -> None:
        super()._setup_model()

        self.policy = MaskedQPolicy(self.observation_space, self.action_space)

    def predict(
        self: SelfMaskedQ,
        observation: np.ndarray,
        deterministic: Optional[bool] = False,
        action_mask: Optional[np.ndarray] = None
    ):
        # Perform ε-greedy action selection
        if not deterministic and np.random.rand() < self.exploration_rate:
            action = self.action_space.sample()
        else:
            if action_mask is None:
                action_mask = self._mask_fn()
                
            action = self.policy.predict(observation=observation, action_mask=action_mask)

        return action
