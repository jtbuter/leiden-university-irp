import gym
import json, os
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

class CustomCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

        self.timestep_rewards = []
        self.episode_returns = []
        self.episode_return = 0

    def _on_step(self) -> bool:
        reward = self.locals['rewards'].item()

        self.episode_return += reward
        self.timestep_rewards.append(reward)

        if self.locals["dones"].item():
            self.episode_returns.append(self.episode_return)

            # print(self.episode_return)

            self.episode_return = 0

        if self.n_calls % 1000 == 0:
            print(self.episode_return)
            # ep_reward = 
            # self.returns.append((sum(env.rewards), env.needs_reset))
            # print(self.n_calls)
            # print(json.dumps(self.locals, indent = 4, default = str))

        return True
    
env = gym.make("CartPole-v1")
monitor = Monitor(env)
custom_callback = CustomCallback()
model = DQN("MlpPolicy", monitor, tensorboard_log = './tensor-logs', learning_rate=0.001)

model.learn(total_timesteps = 150000, log_interval = 1, callback = custom_callback)
# model.save("dqn_cartpole")

# del model # remove to demonstrate saving and loading

# model = DQN.load("dqn_cartpole")

# obs = monitor.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = monitor.step(action)
#     monitor.render()
#     if done:
#       obs = monitor.reset()