import numpy as np
import matplotlib.pyplot as plt

import irp.utils
import irp.envs as envs

from irp.envs.base_env import UltraSoundEnv
from irp.envs.env import Env

image_parameters = {
    'subimage_width': 16,
    'subimage_height': 8,
    'overlap': 0.875
}
neighborhood_parameters = {
    'n_size': 1,
    'overlap': image_parameters['overlap'],
    'neighborhood': 'neumann'
}
agent_parameters = {
    'alpha': 1.0,
    'max_t': 1000,
    'max_e': 1000,
    'ep_max': 1.0,
    'ep_min': 0.05,
    'ep_frac': 0.01,
    'gamma': 0.9,
}

(image, truth), (t_image, t_truth) = irp.utils.read_sample('case10_10.png'), irp.utils.read_sample('case10_11.png')
subimages, sublabels, t_subimages, t_sublabels = irp.utils.extract_subimages(
    image, truth, t_image, t_truth, **image_parameters
)

sample_coord = (192, 248)
sample_id = irp.utils.coord_to_id(sample_coord, image.shape, **image_parameters)

sample, label = subimages[sample_id], sublabels[sample_id]
t_sample, t_label = t_subimages[sample_id], t_sublabels[sample_id]
neighborhood = irp.utils.get_neighborhood_images(
    t_subimages, t_sublabels, sample_id, **dict(image_parameters, **neighborhood_parameters)
)

labels = np.concatenate([neighborhood[1], [t_label]])
bitmasks = []

n_thresholds = 4

for sample, label in zip(*neighborhood):
    intensity_spectrum = envs.utils.get_intensity_spectrum(sample, n_thresholds)
    intensity_spectrum = np.insert(intensity_spectrum, 0, -1.0)

    sample_bitmasks = [
        irp.utils.apply_action_sequence(sample, [th, 0], [envs.utils.apply_threshold, envs.utils.apply_opening]) for th in intensity_spectrum
    ]

    bitmasks.append(sample_bitmasks)

t_intensity_spectrum = envs.utils.get_intensity_spectrum(t_sample, n_thresholds)
t_intensity_spectrum = np.insert(t_intensity_spectrum, 0, -1.0)
t_sample_bitmasks = [
    irp.utils.apply_action_sequence(t_sample, [t_th, 0], [envs.utils.apply_threshold, envs.utils.apply_opening]) for t_th in t_intensity_spectrum
]

bitmasks.append(t_sample_bitmasks)

fig, ax = plt.subplots(nrows=6, ncols=len(intensity_spectrum) + 1, figsize=(15, 15))
# plt.gca().set_axis_off()
# plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
# plt.margins(0,0)

for i, sample_bitmasks in enumerate(bitmasks):
    label = labels[i]
    d_sims, best_d_sim = [], np.inf
    observations = []

    for j, bitmask in enumerate(sample_bitmasks):
        observation = UltraSoundEnv.observation(bitmask)
        observation = [round(obs, 2) for obs in observation]
        d_sim = envs.utils.compute_dissimilarity(label, bitmask)

        d_sims.append(d_sim)
        observations.append(observation)

        best_d_sim = d_sim if d_sim < best_d_sim else best_d_sim

        ax[i][j].title.set_text(f'{observation}')
        ax[i][j].imshow(bitmask, cmap='gray', vmin=0, vmax=1)
        ax[i][j].get_xaxis().set_visible(False)

    for j in range(len(sample_bitmasks)):
        if j in np.argwhere(d_sims == best_d_sim)[0]:
            ax[i][j].title.set_text(f'{observations[j]}*')
        else:
            ax[i][j].title.set_text(f'{observations[j]}')

    ax[i][j + 1].title.set_text(f'Target bitmask')
    ax[i][j + 1].imshow(label, cmap='gray', vmin=0, vmax=1)
    ax[i][j + 1].get_xaxis().set_visible(False)


fig.tight_layout()
plt.show()

# t_environment = Env(t_sample, t_label, n_thresholds=4)
# best_bitmask = irp.utils.apply_action_sequence(sample, [environment.intensity_spectrum[environment.best_ti[0]], 2], [envs.utils.apply_threshold, envs.utils.apply_opening])

# irp.utils.show(best_bitmask, environment.label)

# # irp.utils.show(*bitmasks)
# # irp.utils.show(best_bitmask)

# # Plot test bitmasks
# print(t_environment.best_ti)
# t_bitmasks = [irp.utils.apply_action_sequence(t_sample, [th, 2], [envs.utils.apply_threshold, envs.utils.apply_opening]) for th in t_environment.intensity_spectrum]
# t_best_bitmask = irp.utils.apply_action_sequence(t_sample, [t_environment.intensity_spectrum[t_environment.best_ti[0]], 2], [envs.utils.apply_threshold, envs.utils.apply_opening])

# for t_bitmask in t_bitmasks:
#     print(envs.utils.compute_dissimilarity(t_label, t_bitmask))

# irp.utils.show(t_best_bitmask, t_environment.label)
# # irp.utils.show(*bitmasks)
# # irp.utils.show(best_bitmask)