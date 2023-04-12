from irp.q import Q
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
import numpy as np

class StopOnDone(BaseCallback):
    model: Q

    def __init__(self, verbose: int = 0):
        super().__init__(verbose=verbose)

        self.finished = False
        self._log_freq = 1

    def _on_step(self) -> bool:
        done = self.locals['done']

        if self.finished:
            # Returning False causes rollout collection to stop
            return False

        if self.n_calls % self._log_freq == 0 and self.n_calls > 0 and not done:
            # Collect the last 5 rewards
            recent_rewards = self.model.env.envs[0].rewards[-5:]

            # Write to tensorboard
            self.model._tb_write(
                "episode/running_mean", np.mean(recent_rewards), self.num_timesteps
            )

        # Store if the episode is done, so that in the next time-step we can
        # terminate when this callback is called
        self.finished = done