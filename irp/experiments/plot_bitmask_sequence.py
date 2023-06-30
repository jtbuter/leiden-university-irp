import itertools
from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt

import irp.utils
import irp.envs.utils

from irp.envs.base_env import UltraSoundEnv
from irp.envs.env import Env

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
    'n_thresholds': 7
}

(image, truth), (t_image, t_truth), (p_image, p_truth) = irp.utils.stacked_read_sample('case10_10.png', 'case10_11.png', 'case10_12.png', median_size=15)

# (image, truth), (t_image, t_truth), = irp.utils.read_sample('case10_10.png', median_size=15), irp.utils.read_sample('case10_11.png', median_size=15)
subimages, sublabels, t_subimages, t_sublabels, p_subimages, p_sublabels = irp.utils.extract_subimages(
    image, truth, t_image, t_truth, p_image, p_truth, **image_parameters
)
coords = [(256, 224)]

fig, axes = plt.subplots(nrows=3, ncols=environment_parameters['n_thresholds'] + 2)

for sample_coord in coords:
    x, y = sample_coord
    sample_id = irp.utils.coord_to_id(sample_coord, image.shape, **image_parameters)

    sample, label = subimages[sample_id], sublabels[sample_id]
    intensity_spectrum = irp.envs.utils.get_intensity_spectrum(sample, **environment_parameters, add_minus=True)

    best_dissim, sequence = irp.utils.get_best_dissimilarity(sample, label, [intensity_spectrum, [0]], [irp.envs.utils.apply_threshold, irp.envs.utils.apply_opening], return_seq=True)

    for i, intensity in enumerate(intensity_spectrum):
        bitmask = irp.utils.apply_action_sequence(sample, (intensity, 0), [irp.envs.utils.apply_threshold, irp.envs.utils.apply_opening])
        dissim = irp.envs.utils.compute_dissimilarity(label, bitmask)

        # if i == irp.utils.indexof(sequence[0], intensity_spectrum):
            # axes[0][i].title.set_text('*')

        axes[0][i].title.set_text(round(dissim, 3))
        axes[0][i].imshow(bitmask, cmap='gray', interpolation='none', vmin=0, vmax=1)
        axes[0][i].set_xticks([])
        axes[0][i].set_yticks([])

    axes[0][i + 1].imshow(label, cmap='gray', interpolation='none')
    axes[0][i + 1].set_xticks([])
    axes[0][i + 1].set_yticks([])

    t_sample, t_label = t_subimages[sample_id], t_sublabels[sample_id]
    t_intensity_spectrum = irp.envs.utils.get_intensity_spectrum(t_sample, **environment_parameters, add_minus=True)
    
    best_dissim, sequence = irp.utils.get_best_dissimilarity(t_sample, t_label, [t_intensity_spectrum, [0]], [irp.envs.utils.apply_threshold, irp.envs.utils.apply_opening], return_seq=True)

    for i, intensity in enumerate(t_intensity_spectrum):
        bitmask = irp.utils.apply_action_sequence(t_sample, (intensity, 0), [irp.envs.utils.apply_threshold, irp.envs.utils.apply_opening])

        # if i == irp.utils.indexof(sequence[0], t_intensity_spectrum):
        #     axes[1][i + 1].title.set_text('*')

        axes[1][i].imshow(bitmask, cmap='gray', interpolation='none', vmin=0, vmax=1)
        axes[1][i].set_xticks([])
        axes[1][i].set_yticks([])

    axes[1][i + 1].imshow(t_label, cmap='gray', interpolation='none')
    axes[1][i + 1].set_xticks([])
    axes[1][i + 1].set_yticks([])

    p_sample, p_label = p_subimages[sample_id], p_sublabels[sample_id]
    p_intensity_spectrum = irp.envs.utils.get_intensity_spectrum(p_sample, **environment_parameters, add_minus=True)
    
    best_dissim, sequence = irp.utils.get_best_dissimilarity(p_sample, p_label, [p_intensity_spectrum, [0]], [irp.envs.utils.apply_threshold, irp.envs.utils.apply_opening], return_seq=True)

    for i, intensity in enumerate(p_intensity_spectrum):
        bitmask = irp.utils.apply_action_sequence(p_sample, (intensity, 0), [irp.envs.utils.apply_threshold, irp.envs.utils.apply_opening])

        # if i == irp.utils.indexof(sequence[0], t_intensity_spectrum):
        #     axes[1][i + 1].title.set_text('*')

        axes[2][i].imshow(bitmask, cmap='gray', interpolation='none', vmin=0, vmax=1)
        axes[2][i].set_xticks([])
        axes[2][i].set_yticks([])

    axes[2][i + 1].imshow(p_label, cmap='gray', interpolation='none')
    axes[2][i + 1].set_xticks([])
    axes[2][i + 1].set_yticks([])

    plt.tight_layout()
    plt.show()

# t_intensity_spectrum = irp.envs.utils.get_intensity_spectrum(t_sample, **environment_parameters, add_minus=True)
# best_dissim, sequence = irp.utils.get_best_dissimilarity(t_sample, t_label, [t_intensity_spectrum, [7]], [irp.envs.utils.apply_threshold, irp.envs.utils.apply_opening], return_seq=True)
# bitmask = irp.utils.apply_action_sequence(t_sample, sequence, [irp.envs.utils.apply_threshold, irp.envs.utils.apply_opening])

# print(best_dissim)
# irp.utils.show(bitmask)
