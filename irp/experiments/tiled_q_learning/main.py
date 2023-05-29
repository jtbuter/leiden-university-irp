from typing import Optional
import matplotlib.pyplot as plt
import numpy as np

from irp.experiments.tile_coding.env import Env
from irp.policies.tiled_q_table import TiledQ
from irp.experiments.tiled_q_learning.encoder import Tiled
import irp.wrappers as wrappers
import irp.envs as envs
import irp.utils

def evaluate(environment: Tiled, policy: TiledQ, steps: int = 10, ti: Optional[int] = None):
    ti = (environment.n_thresholds - 1) * (np.random.random() > 0.5)

    state = environment.reset(ti=ti)

    for i in range(steps):
        action = policy.predict(state)
        state, reward, done, info = environment.step(action)

    return info['d_sim'] 

s_width, s_height, overlap, n_size = 16, 8, 0, 1 # Define characteristics for the training and testing samples
subimages, sublabels = irp.utils.get_subimages('case10_10.png', s_width, s_height, overlap) # Get all training instances
sample, label = subimages[1134], sublabels[1134] # Get a specific training instance

test_subimages, test_sublabels = irp.utils.get_subimages('case10_11.png', s_width, s_height, overlap) # Get all training instances
test_sample, test_label = test_subimages[1134], test_sublabels[1134] # Get a specific training instance

n_thresholds, tiles_per_dim, tilings, limits = 4, (2, 2, 2), 64, [(0, 1), (0, 1), (0, 32)]
environment = Env(sample, label, n_thresholds=n_thresholds)
environment = Tiled(environment, tiles_per_dim, tilings, limits)

test_environment = Env(test_sample, test_label, n_thresholds=n_thresholds)
test_environment = Tiled(test_environment, tiles_per_dim, tilings, limits)

alpha = 0.2
gamma = 0.95

t = 0
ep = 1
ep_frac = 0.999
ep_min = 0.3

policy = TiledQ(environment.T.n_tiles, environment.action_space.n, alpha)
exploitation = None
d_sims = []

while t < 5000: # Perform `n` total timesteps
    state = environment.reset()

    for _ in range(15): # Perform 10 timesteps
        if np.random.random() < ep: action = environment.action_space.sample()
        else: action = policy.predict(state)

        next_state, reward, done, info = environment.step(action)
        target = reward + gamma * max(policy.values(next_state))

        policy.update(state, action, target)

        state = next_state

        t += 1

        ep = max(ep_min, ep * ep_frac)

        if exploitation is None and ep == ep_min:
            exploitation = t

    d_sims.append((t, evaluate(test_environment, policy)))

xs, ys = list(zip(*d_sims))

n = 30

plt.plot(xs, np.convolve(ys, np.ones(n) / n, 'same'), label='Dissimilarity')
plt.axhline(test_environment.d_sim, linestyle='-', color='black', label='Optimality')
if exploitation is not None:
    plt.axvline(exploitation, linestyle='--', color='grey')
plt.legend()
plt.show()

# print(test_environment.intensity_spectrum)

# state = test_environment.reset(ti=4)

# for i in range(10):
#     action = policy.predict(state)
#     state, reward, done, info = test_environment.step(action)

#     print(action, info['d_sim'], done)
