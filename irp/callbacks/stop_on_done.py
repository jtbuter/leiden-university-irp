from irp.q import Q
from stable_baselines3.common.callbacks import BaseCallback

class StopOnDoneCallback(BaseCallback):
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

        # Store if the episode is done, so that in the next time-step we can
        # terminate when this callback is called
        self.finished = done