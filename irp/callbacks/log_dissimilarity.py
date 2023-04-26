from __future__ import annotations
import numpy as np
import typing

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean

if typing.TYPE_CHECKING:
    from irp.envs import TrusEnv

class LogDissimilarityCallback(BaseCallback):
    def __init__(self, log_freq: int = 500, verbose: int = 0):
        super().__init__(verbose=verbose)

        self._log_freq = log_freq
        self._dissims = []

    def _on_step(self):
        # Write the dissimilarity at the end of the episode
        if self.locals['done'][0]:
            dissim = self.locals['info'][0]['dissimilarity']
            
            self._dissims.append(dissim)
            
        if self.n_calls % self._log_freq == 0:
            # Write to tensorboard
            self.model._tb_write(
                "rollout//mean_dissimilarity", safe_mean(self._dissims[-100:]), self.num_timesteps
            )

        return True
        
