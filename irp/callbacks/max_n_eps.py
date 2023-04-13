from irp.q import Q
from stable_baselines3.common.callbacks import BaseCallback

class MaxNEpisodesCallback(BaseCallback):
    model: Q

    def __init__(self, max_episodes = None, verbose: int = 0):
        super().__init__(verbose=verbose)

        self.max_episodes = max_episodes

    def _on_step(self) -> bool:
        if self.max_episodes is None:
            return True

        return self.model._episode_num < self.max_episodes