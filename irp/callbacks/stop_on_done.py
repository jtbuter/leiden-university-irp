from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class StopOnDone(BaseCallback):
    def _on_step(self) -> bool:
        done = self.locals['done']

        if done:
            print('Done')

            # Returning False causes rollout collection to stop
            return False
