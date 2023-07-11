from irp.envs.utils import compute_dissimilarity
import os
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics

from irp.experiments.tile_coding.env import Env
from irp.policies.tiled_q_table import TiledQ
from irp.experiments.tiled_q_learning.encoder import Tiled
import irp.utils
import irp

def evaluate(environment: Tiled, policy: TiledQ, steps: int = 10, ti: Optional[int] = None):
    if ti is None:
        ti = (environment.n_thresholds - 1) * (np.random.random() > 0.5)

    state = environment.reset(ti=ti)

    for i in range(steps):
        action = policy.predict(state)
        state, reward, done, info = environment.step(action)

    return info['d_sim']

real = irp.utils.read_image(os.path.join(irp.GIT_DIR, '../data/trus/labels/case10_11.png'))

image_parameters = {
    'subimage_width': 16,
    'subimage_height': 8,
    'overlap': 0
}

(image, label), (t_image, t_label) = irp.utils.stacked_read_sample('case10_10.png', 'case10_11.png')
subimages, sublabels, t_subimages, t_sublabels = irp.utils.extract_subimages(
    image, label, t_image, t_label, **image_parameters
)

n_thresholds, tiles_per_dim, tilings, limits = 4, (2, 2, 2), 64, [(0, 1), (0, 1), (0, 32)] # Characteristics for tile-coding
alpha = 0.2
gamma = 0.95
ep_frac = 0.999
ep_min = 0.3

coords = irp.utils.extract_coordinates(image.shape, **image_parameters)
# coords = [(272, 176), (256, 184), (288, 184), (272, 192)]

result = np.zeros(image.shape)
failed = []

for coord in coords:
    x, y = coord
    index = irp.utils.coord_to_id(coord, image.shape, **image_parameters)

    if not (x >= 192 and x <= 336 and y >= 176 and y <= 288):
        continue

    sample, label = subimages[index], sublabels[index]
    environment = Env(sample, label, n_thresholds=n_thresholds)
    environment = Tiled(environment, tiles_per_dim, tilings, limits)
    
    policy = TiledQ(environment.T.n_features, tilings, environment.action_space.n, alpha)
    
    t = 0
    ep = 1

    while t < 5000: # Perform `n` total timesteps
        state = environment.reset()
        done = False

        if np.random.random() < ep:
            action = environment.action_space.sample()
        else:
            action = policy.predict(state)

        while not done:
            next_state, reward, done, info = environment.step(action)

            if np.random.random() < ep:
                next_action = environment.action_space.sample()
            else:
                next_action = policy.predict(next_state)

            target = reward + gamma * policy.value(next_state, next_action)

            policy.update(state, action, target)

            state = next_state
            action = next_action

            t += 1

            ep = max(ep_min, ep * ep_frac)

    t_sample, t_label = t_subimages[index], t_sublabels[index]
    t_environment = Env(t_sample, t_label, n_thresholds=n_thresholds)
    t_environment = Tiled(t_environment, tiles_per_dim, tilings, limits)

    d_sim = evaluate(t_environment, policy, ti=0)

    print(coord, d_sim, t_environment.d_sim_opt)

    if not np.isclose(d_sim, t_environment.d_sim_opt):
        failed.append(coord)

    result[y:y+image_parameters['subimage_height'], x:x+image_parameters['subimage_width']] = t_environment.bitmask

print(failed)
print(sklearn.metrics.f1_score((real / 255).astype(bool).flatten(), (result / 255).astype(bool).flatten()))

plt.imshow(result, cmap='gray', vmin=0, vmax=1)
plt.show()
