import gymnasium as gym
from heartpole import HeartPole

heartpole = HeartPole(heart_attack_proclivity = 0.1)

from stable_baselines3 import DQN

model = DQN("MlpPolicy", heartpole, verbose=1, policy_kwargs={'net_arch': [16,16]})

model.learn(total_timesteps=500000, log_interval=10)