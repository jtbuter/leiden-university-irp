import os
from typing import Optional
from gym.wrappers.time_limit import TimeLimit
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics

from irp.experiments.tile_coding.env import Env
from irp.policies.tiled_q_table import TiledQ
from irp.experiments.tiled_q_learning.encoder import Tiled
import irp.wrappers as wrappers
import irp.envs as envs
import irp.utils
import irp

def evaluate(environment: Tiled, policy: TiledQ, steps: int = 10, ti: Optional[int] = None, return_done: Optional[bool] = False):
    if ti is None:
        ti = (environment.n_thresholds - 1) * (np.random.random() > 0.5)

    state = environment.reset(ti=ti)

    for i in range(steps):
        action = policy.predict(state)
        state, reward, done, info = environment.step(action)

    if return_done:
        return done

    return info['d_sim']


train, test = 10, 11

real = irp.utils.read_image(os.path.join(irp.GIT_DIR, f'../data/trus/labels/case10_{test}.png'))
s_width, s_height, overlap, n_size = 16, 8, 0, 0 # Define characteristics for the training and testing samples
subimages, sublabels = irp.utils.get_subimages(f'case10_{train}.png', s_width, s_height, overlap) # Get all training instances
t_subimages, t_sublabels = irp.utils.get_subimages(f'case10_{test}.png', s_width, s_height, overlap) # Get all training instances
n_thresholds, tiles_per_dim, tilings, limits = 4, (2, 2, 2), 64, [(0, 1), (0, 1), (0, 32)] # Characteristics for tile-coding
alpha = 0.2
gamma = 0.95
ep_frac = 0
ep_min = 0.1

coords = irp.utils.extract_subimages(np.zeros((512, 512)), s_width, s_height, overlap)[1]

result = np.zeros((512, 512))

failed = []
train_d_sims = []
eval_d_sims = []
exploit = 0
max_e = 5000
max_t = 5000

for coord in coords:
    x, y = coord
    index = irp.utils.coord_to_id(coord, (512, 512), s_width, s_height, overlap)

    if not (x >= 192 and x <= 336 and y >= 176 and y <= 288):
        continue

    t_sample, t_label = t_subimages[index], t_sublabels[index]
    t_environment = Env(t_sample, t_label, n_thresholds=n_thresholds)
    t_environment = Tiled(t_environment, tiles_per_dim, tilings, limits)

    samples, labels = irp.utils.get_neighborhood_images(subimages, sublabels, coord, s_width, s_height, overlap, n_size, (512, 512), 'neumann')
    environments = wrappers.MultiSample([])

    for sample, label in zip(samples, labels):
        environment = Env(sample, label, n_thresholds=n_thresholds)
        environment = TimeLimit(environment, 30)
        environment = Tiled(environment, tiles_per_dim, tilings, limits)

        environments.add(environment)

    environment = environments
    
    policy = TiledQ(environment.T.n_tiles, environment.action_space.n, alpha)
    
    t, e = 0, 0
    ep = 1

    state = environment.reset()

    if np.random.random() < ep: action = environment.action_space.sample()
    else: action = policy.predict(state)

    while e < max_e and t < max_t: # Perform `n` total timesteps
        next_state, reward, done, info = environment.step(action)
        next_action = policy.predict(next_state)
        target = reward + gamma * policy.value(next_state, next_action)

        policy.update(state, action, target)

        state = next_state
        action = next_action

        t += 1

        ep = max(ep_min, ep - ep_frac)

        if done:
            state = environment.reset()

            if np.random.random() < ep: action = environment.action_space.sample()
            else: action = policy.predict(state)
    
            e += 1

    d_sim = evaluate(t_environment, policy, ti=0)

    print(coord, d_sim, t_environment.d_sim)

    if not np.isclose(d_sim, t_environment.d_sim):
        failed.append(coord)

    result[y:y+s_height, x:x+s_width] = t_environment.bitmask

    print(f"Episodes: {e} / {max_e}, timesteps: {t} / {max_t}")

# plt.plot(np.convolve(train_d_sims, np.ones(10) / 10, mode='same'), label='train')
# plt.plot(np.convolve(eval_d_sims, np.ones(10) / 10, mode='same'), label='eval')
# plt.axhline(t_environment.d_sim, linestyle='--', color='red', label='best test d_sim')
# plt.axvline(min(exploit, len(train_d_sims)), linestyle='--', color='grey', label='exploitation')
# plt.legend()
# plt.show()

# irp.utils.show(np.hstack([t_environment.label, t_environment.bitmask]))

# print(failed)
# print(round(1 - (len(failed) / len(coords)), 2), len(failed), len(coords))
print(sklearn.metrics.f1_score(real.astype(bool).flatten(), result.astype(bool).flatten()))

plt.imshow(np.hstack([real, result]), cmap='gray', vmin=0, vmax=1)
plt.show()
