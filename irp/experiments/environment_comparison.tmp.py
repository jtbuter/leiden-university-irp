import os
import numpy as np

from irp.experiments.tile_coding.env import Env as OldEnv
from irp.experiments.tiled_q_learning.encoder import Tiled as OldTiled

from irp.envs.env import Env as NewEnv
from irp.wrappers.tiled import Tiled as NewTiled

import irp.utils
import irp

real = irp.utils.read_image(os.path.join(irp.GIT_DIR, '../data/trus/labels/case10_11.png'))

image_parameters = {
    'subimage_width': 16,
    'subimage_height': 8,
    'overlap': 0
}
environment_parameters = {
    'n_thresholds': 4,
    'opening': 0
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

# coords = irp.utils.extract_coordinates((512, 512), s_width, s_height, 0)
coords = [(272, 176), (256, 184), (288, 184), (272, 192)]

result = np.zeros((512, 512))
failed = []

for coord in coords:
    index = irp.utils.coord_to_id(coord, image.shape, **image_parameters)

    sample, label = subimages[index], sublabels[index]
    
    old_environment = OldEnv(sample, label, n_thresholds=n_thresholds)
    old_environment = OldTiled(old_environment, tiles_per_dim, tilings, limits)

    new_environment = NewEnv(sample, label, n_thresholds=n_thresholds)
    new_environment = NewTiled(new_environment, tiles_per_dim, tilings, limits)

    ti = np.random.randint(old_environment.n_thresholds)

    old_environment.reset(ti=ti)
    new_environment.reset(ti=ti)

    assert np.all(old_environment.bitmask == new_environment.bitmask)

    for i in range(200):
        action = old_environment.action_space.sample()

        old_state, old_reward, old_done, old_info = old_environment.step(action)
        new_state, new_reward, new_done, new_info = new_environment.step(action)

