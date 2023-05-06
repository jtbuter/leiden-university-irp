import gym
import irp.experiments.tile_coding.q as q
import irp.experiments.tile_coding.env as env
import numpy as np
import irp.utils
import irp.envs as envs

# Hyperparameters
parameters = {
    'episodes': 1000,       # Total number of episodes
    'alpha': 0.5,           # Learning rate
    'gamma': 0.9,           # Discount factor
    'epsilon': 1.0,         # Amount of randomness in the action selection
    'epsilon_decay': 0.001  # Fixed amount to decrease
}

sample, label = irp.utils.make_sample_label('case10_10.png', idx=721)[0]

environment = env.Env(sample, label, 6)
qtable = q.learn(environment, parameters)

s = environment.reset(0)

for i in range(10):
    a = np.argmax(qtable[s])
    s, r, d, info = environment.step(a)

    print(r, d, info)