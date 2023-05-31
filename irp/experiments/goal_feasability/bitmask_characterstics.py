import itertools
import os
from matplotlib import pyplot as plt
import numpy as np

import irp
import irp.utils
import irp.envs as envs
import irp.wrappers as wrappers
from irp.envs.ultrasound.ultra_sound_env import UltraSoundEnv

from scipy.ndimage import median_filter

subimage_width, subimage_height = 16, 8
n_size = 1

train_name = f'case10_10.png'
test_name = 'case10_11.png'
subimage_width, subimage_height = 16, 8
overlap = 0.5

# Get all the subimages
train_Xs, train_ys = np.asarray(
    irp.utils.make_sample_label(train_name, width=subimage_width, height=subimage_height, overlap=overlap, idx=None)
)[0]
test_Xs, test_ys = np.asarray(
    irp.utils.make_sample_label(test_name, width=subimage_width, height=subimage_height, overlap=overlap, idx=None)
)[0]

result = np.zeros((512, 512))
real = np.zeros((512, 512))

coord = (256, 264)
idx = irp.utils.coord_to_id(coord, (512, 512), subimage_width, subimage_height, overlap)

train_Xs, train_ys = irp.utils.get_neighborhood_images(
    train_Xs, train_ys, idx, subimage_width, subimage_height, overlap, n_size=n_size, neighborhood='neumann'
)

n_thresholds = 4
markers = {'d': '$D$', 'l': '$L$', 'h': '$H$'}
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection='3d')

for t, (sample, label) in enumerate(zip(train_Xs, train_ys)):
    observations = []
    ths = envs.utils.get_intensity_spectrum(sample, n_thresholds, add_minus=True)
    dissim, seq = irp.utils.get_best_dissimilarity(
        sample, label, actions=[ths, [0, 2, 5]], fns=[envs.utils.apply_threshold, envs.utils.apply_opening], return_seq=True
    )
    
    best_th = seq[0][0]
    best_size = seq[1][0]

    color = next(ax._get_lines.prop_cycler)['color']

    for i, (th, size) in enumerate(itertools.product(ths, [0, 2, 5])):
        marker = []

        bitmask = envs.utils.apply_threshold(sample, th)
        bitmask = envs.utils.apply_opening(bitmask, size)

        a, c, no = UltraSoundEnv.observation(bitmask)

        if th < best_th:
            marker.append(markers['h'])
        elif th > best_th:
            marker.append(markers['l'])
        else:
            marker.append(markers['d'])

        if size < best_size:
            marker.append(markers['h'])
        elif size > best_size:
            marker.append(markers['l'])
        else:
            marker.append(markers['d'])

        if size == best_size and th == best_th:
            ax.scatter(a, c, no, depthshade=False, s=300, color=color, marker=' '.join(marker), label=f'{t}' if i == 0 else None)
        else:
            ax.scatter(a, c, no, depthshade=False, s=100, color=color, marker=' '.join(marker), label=f'{t}' if i == 0 else None)

ax.set_xlabel('area')
ax.set_ylabel('compactness')
ax.set_zlabel('objects')

plt.legend()
plt.show()
