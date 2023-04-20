from stable_baselines3.common.callbacks import BaseCallback

import numpy as np

class ActionDiversityCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose=verbose)

        self._actions = []

    def _on_step(self):
        action = self.locals['action'][0]

        self._actions.append(action)

        return True

    def _on_training_end(self) -> None:
        print(np.unique(self._actions, return_counts=True))