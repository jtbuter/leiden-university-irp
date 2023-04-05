from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class CustomCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

        self.rewards = []
        self.total_episode_reward = 0

    def _on_step(self) -> bool:
        reward, done = self.locals['reward'], self.locals['done']

        self.total_episode_reward += reward

        if done:
            self.rewards.append(self.total_episode_reward)
            self.total_episode_reward = 0

        if self.n_calls % 1000 == 0:
            print(self.n_calls, np.mean(self.rewards[-100:]))