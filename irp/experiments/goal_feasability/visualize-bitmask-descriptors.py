import itertools
import os
from matplotlib import pyplot as plt
import numpy as np

import irp
import irp.utils
import irp.envs as envs
import irp.wrappers as wrappers
from irp.envs.ultrasound.ultra_sound_env import UltraSoundEnv

train_name = 'case10_10.png'
test_name = 'case10_11.png'
subimage_width, subimage_height = 8, 8
overlap = 0.0

# Get all the subimages
(train_Xs, train_ys), (test_Xs, test_ys) = irp.utils.make_sample_label(
    train_name, test_name, width=subimage_width, height=subimage_height, overlap=overlap, idx=None
)


n_thresholds = 5
n_size = 1
coordinate = (272, 216)
train_Xs, train_ys = irp.utils.get_neighborhood_images(
    train_Xs, train_ys, coordinate, subimage_width, subimage_height, overlap, n_size=n_size, neighborhood=np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ], dtype=bool)
)

markers = ['$D$', '$L$', '$H$']
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection='3d')

lower = []
higher = []
perfect = []

for t, (sample, label) in enumerate(zip(train_Xs, train_ys)):
    print(t)
    observations = []
    ths = envs.utils.get_intensity_spectrum(sample, n_thresholds, add_minus=True)
    dissim, (best_th,) = irp.utils.get_best_dissimilarity(sample, label, [ths], [envs.utils.apply_threshold], return_seq=True)

    print(dissim, best_th)

    color = next(ax._get_lines.prop_cycler)['color']

    for i, th in enumerate(ths[1:]):
        bitmask = envs.utils.apply_threshold(sample, th)
        # bitmask = envs.utils.apply_opening(bitmask, 1)
        a, c, no = UltraSoundEnv.observation(bitmask)

        if th < best_th:
            marker = markers[1]
        elif th > best_th:
            marker = markers[2]
        else:
            marker = markers[0]

        ax.scatter(a, c, no, depthshade=False, s=40, color=color, marker=marker, label=f'{t}' if i == 0 else None)

idx = irp.utils.coord_to_id(coordinate, (512, 512), subimage_width, subimage_height, overlap)
sample = test_Xs[idx]
label = test_ys[idx]

ths = envs.utils.get_intensity_spectrum(sample, n_thresholds, add_minus=True)

dissim, (best_th,) = irp.utils.get_best_dissimilarity(sample, label, [ths], [envs.utils.apply_threshold], return_seq=True)
print(dissim)
color = next(ax._get_lines.prop_cycler)['color']

for i, th in enumerate(ths):
    bitmask = envs.utils.apply_threshold(sample, th)
    a, c, no = UltraSoundEnv.observation(bitmask)

    if th < best_th:
        marker = markers[1]
    elif th > best_th:
        marker = markers[2]
    else:
        marker = markers[0]

    ax.scatter(a, c, no, depthshade=False, s=100, color=color, marker=marker, label=f'Test' if i == 0 else None)

ax.set_xlabel('area')
ax.set_ylabel('compactness')
ax.set_zlabel('objects')

plt.legend()
plt.show()
