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
    'episodes': 500,       # Total number of episodes
    'alpha': 0.5,           # Learning rate
    'gamma': 0.9,           # Discount factor
    'epsilon': 1.0,         # Amount of randomness in the action selection
    'epsilon_decay': 0.001  # Fixed amount to decrease
}

idx = 721
sample, label = irp.utils.make_sample_label('case10_10.png', idx=idx)[0]
n_thresholds = 6

# Compute the best similarty we can obtain
best_d_sim = irp.utils.get_best_dissimilarity(sample, label, n_thresholds)

print('Best possible dissimilarity:', best_d_sim)

bins = (5, 5, 4)

environment = env.Env(sample, label, n_thresholds)
environment = wrappers.Discretize(environment, (0, 0, 1), (1, 1, bins[2]), bins)
success = np.zeros((10,))

for i in range(success.size):
    qtable, epsilons = q.learn(environment, parameters)

    # Assert that we succeeded in learning
    success[i] = True

    s = tuple(environment.reset(ti=n_thresholds - 1))

    for j in range(10):
        a = np.argmax(qtable[s])
        s, r, d, info = environment.step(a)
        s = tuple(s)

    success[i] = np.isclose(info['d_sim'], best_d_sim).astype(int)

print(f'Success rate: {((success.sum() / success.size) * 100):.2f} %')