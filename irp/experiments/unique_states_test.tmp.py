import itertools
import numpy as np
import matplotlib.pyplot as plt

import irp.utils
import irp.envs.utils

from irp.envs.base_env import UltraSoundEnv
from irp.envs.sahba_env import Env
from irp.wrappers.tiled import Tiled

image_parameters = {
    'subimage_width': 32,
    'subimage_height': 16,
    'overlap': 0.5
}
neighborhood_parameters = {
    'n_size': 0,
    'overlap': image_parameters['overlap'],
    'neighborhood': 'neumann'
}
environment_parameters = {
    'n_thresholds': 6,
    'openings': [0, 2, 5]
}
tiling_parameters = {
    'tiles_per_dim': (4, 4, 2),
    'tilings': 16,
    'limits': [(0, 1), (0, 1), (0, 4)]
}

(image, truth), (t_image, t_truth), (p_image, p_truth) = irp.utils.stacked_read_sample('case10_10.png', 'case10_11.png', 'case10_12.png', median_size=7)
subimages, sublabels, t_subimages, t_sublabels, p_subimages, p_sublabels = irp.utils.extract_subimages(
    image, truth, t_image, t_truth, p_image, p_truth, **image_parameters
)

coord = (256, 176)
sample_id = irp.utils.coord_to_id(coord, image.shape, **image_parameters)
sample, label = subimages[sample_id], sublabels[sample_id]

environment = Env(sample, label, **environment_parameters)
tiled = Tiled(environment, **tiling_parameters)

unique_bitmasks = set()
unique_states = set()

for intensity in irp.envs.utils.get_intensity_spectrum(sample, environment_parameters['n_thresholds'], add_minus=True):
    for vj in environment_parameters['openings']:
        bitmask = irp.utils.apply_action_sequence(sample, (intensity, vj), (irp.envs.utils.apply_threshold, irp.envs.utils.apply_opening))
        flat_bitmask = tuple(bitmask.flatten())
        state = tuple(tiled.encode(tiled.T, UltraSoundEnv.observation(bitmask)))

        unique_bitmasks.add(flat_bitmask)
        unique_states.add(state)

print(len(unique_states), len(unique_bitmasks))

