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
    
    print(info['d_sim'], policy.values(state))

    for i in range(steps):

        action = policy.predict(state, environment.action_mask, deterministic=True)
        state, reward, done, info = environment.step(action)
        
        print(info['d_sim'], policy.values(state))

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
    'n_thresholds': 4,
    'opening': 0
}

(image, label), (t_image, t_label) = irp.utils.stacked_read_sample('case10_10.png', 'case10_11.png')
subimages, sublabels, t_subimages, t_sublabels = irp.utils.extract_subimages(
    image, label, t_image, t_label, **image_parameters
)

coords = [(272, 176), (256, 184), (288, 184), (272, 192)]
main_area_start = None

for coord in coords:
    sample_id = irp.utils.coord_to_id(coord, image.shape, **image_parameters)
    sample, label = subimages[sample_id], sublabels[sample_id]

    environment = Env(sample, label, **environment_parameters)
    environment = Tiled(environment, **tiling_parameters)

    agent = Sarsa(environment)
    policy = agent.learn(**agent_parameters)

    d_sim = evaluate(environment, policy, ti=0)

    print(coord, d_sim, environment.d_sim_opt)
