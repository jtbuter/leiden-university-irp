from irp.q import Q
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean
import numpy as np

class LogNStepsCallback(BaseCallback):
    model: Q

    def __init__(self, freq: int = 1, verbose: int = 0):
        super().__init__(verbose=verbose)

        self.finished = False
        self._log_freq = freq
        self.rewards = []

    def _on_step(self) -> bool:
        self.rewards.append(self.locals['reward'].item())

        if self.n_calls % self._log_freq == 0:
            # Collect the last n rewards
            recent_rewards = self.rewards[-100:]

            # Write to tensorboard
            self.model._tb_write(
                "rollout/running_mean", safe_mean(recent_rewards), self.num_timesteps
            )