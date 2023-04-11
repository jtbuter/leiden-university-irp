from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
import numpy as np

class StopOnDone(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose=verbose)

        self.finished = False
        self._log_freq = 1

    def _on_step(self) -> bool:
        done = self.locals['done']

        if self.finished:
            # Returning False causes rollout collection to stop
            return False

        # if self.n_calls % self._log_freq == 0 and self.n_calls > 0 and not done:
        #     self.tb_formatter.writer.add_scalar(
        #         "episode/running_average",
        #         np.mean(self.model.env.envs[0].rewards[-5:]),
        #         self.num_timesteps
        #     )

        #     self.tb_formatter.writer.flush()

        # Store if the episode is done, so that in the next time-step we can
        # terminate when this callback is called
        self.finished = done

    def _on_training_start(self):
        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))
