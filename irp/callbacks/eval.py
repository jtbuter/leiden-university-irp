from irp.q import Q
import irp.utils
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class EvalCallback(BaseCallback):
    model: Q

    def __init__(self, eval_env, n_eval_episodes: int = 20, eval_freq: int = 10, verbose: int = 0):
        super().__init__(verbose=verbose)

        self._eval_env = eval_env
        self._log_freq = eval_freq
        self._n_eval_episodes = n_eval_episodes

    def _on_step(self) -> bool:
        if self.n_calls % self._log_freq == 0 and self.n_calls > 0:
            mean_reward, std_reward = irp.utils.evaluate_policy(
                self.model, self._eval_env, n_eval_episodes=self._n_eval_episodes
            )

            self.model.logger.record("eval/mean_reward", float(mean_reward))
            self.model.logger.dump(self.model.num_timesteps)