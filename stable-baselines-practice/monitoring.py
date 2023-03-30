import gym
import json, os
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

class CustomCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

        self.returns = []

    def _on_step(self) -> bool:
        # env = self.locals['self'].env.envs[0]

        print(self.locals['rewards'].item())        

        # if self.locals["dones"].item():
        #     print('x')

        # print(env.episode_returns)

        # if self.n_calls % 10000 == 0:
            # ep_reward = 
            # self.returns.append((sum(env.rewards), env.needs_reset))
            # print(self.n_calls)
            # print(json.dumps(self.locals, indent = 4, default = str))

        return True
    
env = gym.make("CartPole-v1")
monitor = Monitor(env)
custom_callback = CustomCallback()
model = DQN("MlpPolicy", monitor)

model.learn(total_timesteps = 250000, callback = custom_callback)
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