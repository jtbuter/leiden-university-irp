from env import UltraSoundEnv
import tensorflow as tf
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO, DQN
import stable_baselines3.common.env_checker
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

import gym
import utils

import time
import random
import os
import numpy as np

log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)

image = cv.imread("../data/trus/images/case10_11.png", cv.IMREAD_GRAYSCALE)
label = cv.imread("../data/trus/labels/case10_11.png", cv.IMREAD_GRAYSCALE)
subimage = utils.create_subimages(image, rows = 32, cols = 16)[0][184]
sublabel = utils.create_subimages(label, rows = 32, cols = 16)[0][184]

env = UltraSoundEnv(subimage, sublabel)

tmp_path = "tmp/sb3_log/"
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

model = DQN(
    env = env,
    policy = "MlpPolicy",
    learning_rate = 1e-4,
    buffer_size = 10000,
    exploration_fraction = 0.1,
    exploration_final_eps = 0.01,
    train_freq = 4,
    learning_starts = 1000,
    target_update_interval = 1000,
    verbose = 1
)
model.set_logger(new_logger)
model.learn(total_timesteps = 25000, log_interval = 4)

# obs = env.reset()

# for i in range(1000):
#     action, _state = model.predict(obs)
#     obs, reward, done, truncated, info = env.step(action)
    
#     env.render()

#     if done:
#         obs = env.reset()

#         print('Done')


# print(env.dissim)


# while 1:
#     state, reward, done, info = env.step(env.action_space.sample())

#     print(env.dissim)
#     print(env._compute_reward(sublabel))

#     env.render()
