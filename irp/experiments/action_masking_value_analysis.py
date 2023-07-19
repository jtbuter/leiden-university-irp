import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics

import irp.utils

from irp.envs.env import Env
from irp.agents.sarsa import Sarsa
from irp.agents.qlearning import Qlearning
from irp.wrappers.tiled import Tiled
from irp.wrappers.masking import ActionMasker

def evaluate(environment: Tiled, policy, steps: int = 10, ti: int = None):
    # Keep state-value-similarity-config history
    histories = []

    state, info = environment.reset(ti=ti)
    
    histories.append((np.max(policy.values(state)), info['d_sim'], info['configuration'], environment.bitmask))

    print(info['d_sim'], policy.values(state))

    for i in range(steps):
        action = policy.predict(state, environment.action_mask, deterministic=True)
        state, reward, done, info = environment.step(action)
    
        histories.append((np.max(policy.values(state)), info['d_sim'], info['configuration'], environment.bitmask))
        
        print(info['d_sim'], policy.values(state))

    values, similarities, configs, bitmasks = [np.asarray(history) for history in zip(*histories)]
    path = irp.utils.find_repeating_path(configs)

    # There was a repeating path
    if path:
        best_bitmask = bitmasks[np.argmin(values[path]) + path[0]]
        best_d_sim = similarities[np.argmin(values[path]) + path[0]]

        return best_bitmask, best_d_sim

    # Find the best guess of the optimal dissimilarity
    return bitmasks[np.argmin(values)], similarities[np.argmin(values)]

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
    'opening': [0],
    'sahba': False
}

(image, truth), (t_image, t_truth) = irp.utils.stacked_read_sample('case10_10.png', 'case10_11.png')
subimages, sublabels, t_subimages, t_sublabels = irp.utils.extract_subimages(
    image, truth, t_image, t_truth, **image_parameters
)

coords = [(272, 176), (256, 184), (288, 184), (272, 192)]

for coord in coords:
    sample_id = irp.utils.coord_to_id(coord, image.shape, **image_parameters)
    sample, label = subimages[sample_id], sublabels[sample_id]
    t_sample, t_label = t_subimages[sample_id], t_sublabels[sample_id]

    environment = Env(sample, label, **environment_parameters)
    environment = Tiled(environment, **tiling_parameters)

    agent = Sarsa(environment)
    policy = agent.learn(**agent_parameters)

    t_environment = Env(t_sample, t_label, **environment_parameters)
    t_environment = Tiled(t_environment, **tiling_parameters)

    bitmask, d_sim = evaluate(t_environment, policy, ti=0)

    print(coord, d_sim, t_environment.d_sim_opt)
