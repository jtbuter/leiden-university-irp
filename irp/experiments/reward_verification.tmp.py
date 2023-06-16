import matplotlib.pyplot as plt
import numpy as np

import irp
import irp.utils
import irp.envs as envs

from irp.envs.env import Env
from irp.envs.base_env import UltraSoundEnv
from irp.wrappers.masking import ActionMasker
from irp.wrappers.tiled import Tiled
from irp.agents.qlearning import Qlearning

results = []

def eval(local):
    d_sim, done, bitmask = irp.utils.evaluate(environment, local['self'].policy, max_steps=environment.n_thresholds)

    results.append(d_sim)

callback = {
    'interval': 10,
    'callback': eval
}

image_parameters = {
    'subimage_width': 32,
    'subimage_height': 16,
    'overlap': 0.5
}
neighborhood_parameters = {
    'n_size': 1,
    'overlap': image_parameters['overlap'],
    'neighborhood': 'neumann'
}
tiling_parameters = {
    'tiles_per_dim': (2, 2, 1),
    'tilings': 16,
    'limits': [(0, 1), (0, 1), (0, 4)]
}
agent_parameters = {
    'alpha': 0.8,
    'max_t': np.inf,
    'max_e': 2000,
    'eps_max': 0.6,
    'eps_min': 0.6,
    'eps_frac': 0.001,
    'gamma': 0.6,
}
environment_parameters = {
    'n_thresholds': 8,
    'opening': 8
}

(image, truth), (t_image, t_truth) = irp.utils.read_sample('case10_10.png'), irp.utils.read_sample('case10_11.png')
subimages, sublabels, t_subimages, t_sublabels = irp.utils.extract_subimages(
    image, truth, t_image, t_truth, **image_parameters
)
coord = (288 - 16, 272 + 8)

sample_id = irp.utils.coord_to_id(coord, image.shape, **image_parameters)
sample, label = subimages[sample_id], sublabels[sample_id]
t_sample, t_label = t_subimages[sample_id], t_sublabels[sample_id]

environment = Env(sample, label, **environment_parameters)
environment = Tiled(environment, **tiling_parameters)
environment = ActionMasker(environment)

for i in range(environment.n_thresholds):
    reward1, done1, info1, reward2, done2, info2 = (None,) * 6

    state, info = environment.reset(ti=i)

    if i != 0:
        state, reward1, done1, info1 = environment.step(0)

    if i != environment.n_thresholds - 1:
        state, reward2, done2, info2 = environment.step(1)

    print('left', reward1, done1, info1, 'right', reward2, done2, info2)
    # environment.step(1)