import os
from typing import Optional
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

real = irp.utils.read_image(os.path.join(irp.GIT_DIR, '../data/trus/labels/case10_11.png'))

s_width, s_height, overlap, n_size = 16, 8, 0.875, 0 # Define characteristics for the training and testing samples
subimages, sublabels = irp.utils.get_subimages('case10_10.png', s_width, s_height, overlap) # Get all training instances
t_subimages, t_sublabels = irp.utils.get_subimages('case10_11.png', s_width, s_height, overlap) # Get all training instances
n_thresholds, tiles_per_dim, tilings, limits = 4, (2, 2, 2), 64, [(0, 1), (0, 1), (0, 32)] # Characteristics for tile-coding
alpha = 0.2
gamma = 0.95
ep_frac = 0
ep_min = 0.1

coords = [(256, 176), (288, 176), (240, 184), (256, 184), (224, 192), (240, 192), (304, 192), (224, 200), (240, 200), (304, 200), (320, 200), (224, 208), (240, 208), (304, 208), (320, 208), (208, 216), (240, 216), (208, 224), (336, 224), (208, 232), (192, 240), (336, 240), (192, 248), (272, 248), (304, 248), (208, 264), (256, 264), (272, 264), (208, 272), (224, 272), (240, 272), (256, 272), (288, 272), (240, 280), (288, 280), (304, 280), (304, 288), (320, 288)]

result = np.zeros((512, 512))
real = np.zeros((512, 512))

failed = []
train_d_sims = []
eval_d_sims = []
exploit = 0

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
        environment = Tiled(environment, tiles_per_dim, tilings, limits)
        environments.add(environment)

    environment = environments
    
    policy = TiledQ(environment.T.n_tiles, environment.action_space.n, alpha)
    
    t = 0
    ep = 1

    while t < 5000: # Perform `n` total timesteps
        state = environment.reset()
        if np.random.random() < ep: action = environment.action_space.sample()
        else: action = policy.predict(state)

        for _ in range(30): # Perform `k` timesteps
            next_state, reward, done, info = environment.step(action)
            next_action = policy.predict(next_state)
            target = reward + gamma * policy.value(next_state, next_action)

            policy.update(state, action, target)

            state = next_state
            action = next_action

            t += 1

            ep = max(ep_min, ep - ep_frac)

            if exploit <= 0 and ep == ep_min:
                exploit = t

        # train_d_sims.append(evaluate(environment, policy, return_done=True))
        # eval_d_sims.append(evaluate(t_environment, policy, ti=0))

    d_sim = evaluate(t_environment, policy, ti=0)

    print(coord, d_sim, t_environment.d_sim)

    if not np.isclose(d_sim, t_environment.d_sim):
        failed.append(coord)

    result[y:y+s_height, x:x+s_width] = t_environment.bitmask
    real[y:y+s_height, x:x+s_width] = t_environment.label

# plt.plot(np.convolve(train_d_sims, np.ones(10) / 10, mode='same'), label='train')
# plt.plot(np.convolve(eval_d_sims, np.ones(10) / 10, mode='same'), label='eval')
# plt.axhline(t_environment.d_sim, linestyle='--', color='red', label='best test d_sim')
# plt.axvline(min(exploit, len(train_d_sims)), linestyle='--', color='grey', label='exploitation')
# plt.legend()
# plt.show()

# irp.utils.show(np.hstack([t_environment.label, t_environment.bitmask]))

print(failed)
print(round(1 - (len(failed) / len(coords)), 2), len(failed), len(coords))
# print(sklearn.metrics.f1_score((real / 255).astype(bool).flatten(), (result / 255).astype(bool).flatten()))

# plt.imshow(np.hstack([real, result]), cmap='gray', vmin=0, vmax=1)
# plt.show()
