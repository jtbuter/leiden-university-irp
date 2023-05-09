import gym
import irp.experiments.tile_coding.q as q
import irp.experiments.tile_coding.env as env
import numpy as np
import irp.utils
import irp.envs as envs
import matplotlib.pyplot as plt
import irp.wrappers as wrappers

# Hyperparameters
parameters = {
    'learning_delay': 0,    # Delay until epsilon starts updating
    'episodes': 500,        # Total number of episodes
    'alpha': 0.5,           # Learning rate
    'gamma': 0.9,           # Discount factor
    'epsilon': 1.0,         # Amount of randomness in the action selection
    'epsilon_decay': 0.001, # Fixed amount to decrease
    'tilings': 16           # Number of tilings to use
}

idx = 721
sample, label = irp.utils.make_sample_label('case10_10.png', idx=idx)[0]
n_thresholds = 6

iht = wrappers.utils.IHT(2048)
tilings = parameters['tilings']

environment = env.Env(sample, label, n_thresholds)
environment = wrappers.Tiled(environment, lows=(0, 0, 1), highs=(1, 1, 32), tilings=tilings, iht=iht)

for i in range(n_thresholds):
    print(environment.reset(ti=i))