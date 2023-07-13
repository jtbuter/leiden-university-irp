import numpy as np

import irp.utils

from irp.wrappers.tiled import Tiled
from irp.envs.ranged_env import RangedEnv

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
    'eps_max': 0.8,
    'eps_min': 0.8,
    'eps_frac': 1.0,
    'gamma': 0.8,
}
environment_parameters = {
    'n_thresholds': 5,
    'opening': 0,
    'sahba': True
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

    environment = RangedEnv(sample, label, **environment_parameters)

    environment.reset(ti=(0, 0))

    print(np.asarray(environment.action_mapping)[environment.action_mask()])
