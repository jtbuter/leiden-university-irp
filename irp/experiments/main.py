import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics

import irp.utils

from irp.envs.env import Env
from irp.agents.sarsa import Sarsa
from irp.wrappers.tiled import Tiled
from irp.wrappers.masking import ActionMasker

def evaluate(environment: Tiled, policy, steps: int = 10, ti: int = None):
    state, info = environment.reset(ti=ti)

    for i in range(steps):
        action = policy.predict(state, environment.action_mask, deterministic=True)
        state, reward, done, info = environment.step(action)

    return info['d_sim']

image_parameters = {
    'subimage_width': 16,
    'subimage_height': 8,
    'overlap': 0
}
tiling_parameters = {
    'tiles_per_dim': (2, 2, 2),
    'tilings': 64,
    'limits': [(0, 1), (0, 1), (0, 32)]
}
agent_parameters = {
    'alpha': 0.2,
    'max_t': 5000,
    'max_e': np.inf,
    'eps_max': 1.0,
    'eps_min': 0.3,
    'eps_frac': 0.999,
    'gamma': 0.95,
}
environment_parameters = {
    'n_thresholds': 5,
    'opening': 0
}

real = irp.utils.read_image(os.path.join(irp.GIT_DIR, '../data/trus/labels/case10_11.png'))

(image, truth), (t_image, t_truth) = irp.utils.read_sample('case10_10.png'), irp.utils.read_sample('case10_11.png')
subimages, sublabels, t_subimages, t_sublabels = irp.utils.extract_subimages(
    image, truth, t_image, t_truth, **image_parameters
)

result = np.zeros(image.shape)
coords = irp.utils.extract_coordinates(image.shape, **image_parameters)
main_area_start = None

for coord in coords:
    x, y = coord

    # Don't waste processing power for now (TODO: dit verwijderen voor uiteindelijke report resultaten)
    if not (x >= 192 and x <= 336 and y >= 176 and y <= 288):
        continue

    sample_id = irp.utils.coord_to_id(coord, image.shape, **image_parameters)
    sample, label = subimages[sample_id], sublabels[sample_id]

    environment = Env(sample, label, **environment_parameters)
    environment = Tiled(environment, **tiling_parameters)
    # environment = ActionMasker(environment)

    agent = Sarsa(environment)
    policy = agent.learn(**agent_parameters)

    t_sample, t_label = t_subimages[sample_id], t_sublabels[sample_id]
    t_environment = Env(t_sample, t_label, **environment_parameters)
    t_environment = Tiled(t_environment, **tiling_parameters)
    # t_environment = ActionMasker(t_environment)

    d_sim = evaluate(t_environment, policy, ti=0)

    print(coord, d_sim, t_environment.d_sim_opt)

    result[y:y+image_parameters['subimage_height'], x:x+image_parameters['subimage_width']] = t_environment.bitmask

print(sklearn.metrics.f1_score((real / 255).astype(bool).flatten(), (result / 255).astype(bool).flatten()))

plt.imshow(result, cmap='gray', vmin=0, vmax=1)
plt.show()
