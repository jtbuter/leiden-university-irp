import itertools
import numpy as np
import matplotlib.pyplot as plt

import irp.utils
import irp.envs.utils

from irp.envs.base_env import UltraSoundEnv
from irp.envs.env import Env

image_parameters = {
    'subimage_width': 32,
    'subimage_height': 16,
    'overlap': 1 - (1 / 4)
}
neighborhood_parameters = {
    'n_size': 0,
    'overlap': image_parameters['overlap'],
    'neighborhood': 'neumann'
}
environment_parameters = {
    'n_thresholds': 6
}

(image, truth), (t_image, t_truth) = irp.utils.read_sample('case10_10.png'), irp.utils.read_sample('case10_11.png')
subimages, sublabels, t_subimages, t_sublabels = irp.utils.extract_subimages(
    image, truth, t_image, t_truth, **image_parameters
)
coords = irp.utils.extract_coordinates(image.shape, **dict(image_parameters, **{'overlap': 0}))

result = np.zeros((512, 512))

for sample_coord in coords:
    x, y = sample_coord

    sample_id = irp.utils.coord_to_id(sample_coord, image.shape, **image_parameters)
    sample, label = subimages[sample_id], sublabels[sample_id]
    intensity_spectrum = irp.envs.utils.get_intensity_spectrum(sample, **environment_parameters, add_minus=True)

    dissim, sequence = irp.utils.get_best_dissimilarity(sample, label, [itertools.product(intensity_spectrum, intensity_spectrum), [8]], [irp.envs.utils.apply_threshold, irp.envs.utils.apply_opening], return_seq=True)
    bitmask = irp.utils.apply_action_sequence(sample, sequence, [irp.envs.utils.apply_threshold, irp.envs.utils.apply_opening])

    result[y:y+image_parameters['subimage_height'], x:x+image_parameters['subimage_width']] = bitmask

print(irp.utils.dice(truth, result))
irp.utils.show(result)

