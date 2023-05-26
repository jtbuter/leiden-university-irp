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

# train_name = f'case10_{6}.png'
# image = irp.utils.read_image(os.path.join(irp.GIT_DIR, f'../data/trus/images/{train_name}'))

# image = irp.utils.kuwahara(image, method='gaussian', radius=3)    # default sigma: computed by OpenCV

# irp.utils.show(image)

# id_to_loc_map = [
#     'bottom-left', 'bottom-center', 'bottom-right',
#     'center-left', 'middle', 'center-right',
#     'top-left', 'top-center', 'top-right',
# ]

subimage_width, subimage_height = 16, 8
overlap = 0
n_size = 0

# image = median_filter(image, 3)

# # plt.figure(figsize=(15, 15))
# # irp.utils.show(image)

# # raise Exception


# for i in range(5, 17):
    # print(i)

    # # Train and test filename, and characteristics of the subimages
    # train_name = f'case10_{i}.png'
    # real = irp.utils.read_image(os.path.join(irp.GIT_DIR, f'../data/trus/labels/{train_name}'))
    # coords = irp.utils.extract_subimages(real, subimage_width, subimage_height)[1]

    # # Get all the subimages
    # train_Xs, train_ys = np.asarray(
    #     irp.utils.make_sample_label(train_name, width=subimage_width, height=subimage_height, overlap=overlap, idx=None)
    # )[0]

    # result = np.zeros((512, 512))

    # n_thresholds = 5

    # for sample, label, coord in zip(train_Xs, train_ys, coords):
    #     x, y = coord
    #     ths = envs.utils.get_intensity_spectrum(sample, n_thresholds, add_minus=True)

    #     dissim, seq = irp.utils.get_best_dissimilarity(sample, label, [ths], [envs.utils.apply_threshold], return_seq=True)

    #     bitmask = envs.utils.apply_threshold(sample, seq[0])

    #     result[y:y+subimage_height, x:x+subimage_width] = bitmask

    # recall = irp.utils.area_of_overlap(real, result)
    # precision = irp.utils.precision(real, result)
    # f1 = irp.utils.f1(real, result)
    # jaccard = irp.utils.jaccard(real, result)
    # dice = irp.utils.dice(real, result)

    # plt.figure(figsize=(15, 15))
    # plt.title(f'recall: {round(recall, 2)}, precision: {round(precision, 2)}, jaccard: {round(jaccard, 2)}, f1: {round(f1, 2)}, dice: {round(dice, 2)}')
    # irp.utils.show(np.hstack([result, real]))

    # diffs[i - 5, :] = abs(np.asarray(recall) - np.asarray(precision))

    # ax.plot(list(range(10, 20)), recall, label=f'{i} recall', linestyle='--', color=color)
    # ax.plot(list(range(10, 20)), precision, label=f'{i} precision', linestyle='-', color=color)

        # plt.figure(figsize=(15, 15))
        # plt.title(
        #     f'Recall: {round(irp.utils.area_of_overlap(real, result) * 100, 2)}%,' + \
        #     f'Precision: {round(irp.utils.precision(real, result) * 100, 2)}%, {i}'
        # )
        # irp.utils.show(np.hstack([result, real]))

# raise Exception

train_name = f'case10_{6}.png'
test_name = 'case10_7.png'
subimage_width, subimage_height = 8, 4
overlap = 0.75

# Get all the subimages
train_Xs, train_ys = np.asarray(
    irp.utils.make_sample_label(train_name, width=subimage_width, height=subimage_height, overlap=overlap, idx=None)
)[0]
test_Xs, test_ys = np.asarray(
    irp.utils.make_sample_label(test_name, width=subimage_width, height=subimage_height, overlap=overlap, idx=None)
)[0]

result = np.zeros((512, 512))
real = np.zeros((512, 512))
coords = irp.utils.extract_subimages(irp.utils.read_image(os.path.join(irp.GIT_DIR, f'../data/trus/images/{train_name}')), subimage_width, subimage_height, overlap)[1]

coord = (272, 212)

# raise Exception

# coordinates = [(256, 176), coord, (240, 184), (224, 192)]
# ids = np.asarray([
#     irp.utils.coord_to_id(coord, (512, 512), subimage_width, subimage_height, overlap) for coord in coordinates
# ])
# 
# train_Xs = train_Xs[ids]
# train_ys = train_ys[ids]

n_size = 1
idx = irp.utils.coord_to_id(coord, (512, 512), subimage_width, subimage_height, overlap)
train_Xs, train_ys = irp.utils.get_neighborhood_images(
    train_Xs, train_ys, idx, subimage_width, subimage_height, overlap, n_size=n_size, neighborhood='neumann'
)

# sample, label = train_Xs[2], train_ys[2]
# ths = envs.utils.get_intensity_spectrum(sample, n_thresholds, add_minus=True)

# dissim, seq = irp.utils.get_best_dissimilarity(sample, label, [itertools.product(ths, ths), [0, 2, 5, 7, 9, 13]], [envs.utils.apply_threshold, envs.utils.apply_opening], return_seq=True)

# bitmask = envs.utils.apply_threshold(sample, *seq[0])
# bitmask = envs.utils.apply_opening(bitmask, 0)

# print(dissim, seq)

# irp.utils.show(bitmask)
# irp.utils.show(np.hstack([label]))


# print(len(train_Xs))

# # irp.utils.show(train_ys[4])

n_thresholds = 4
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
    # ths = envs.utils.get_intensity_spectrum(sample, n_thresholds, add_minus=True)
    dissim, (best_th,) = irp.utils.get_best_dissimilarity(sample, label, [ths], [envs.utils.apply_threshold], return_seq=True)
    print(dissim)

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

        # no = 1

        ax.scatter(a, c, no, depthshade=False, s=40, color=color, marker=marker, label=f'{t}' if i == 0 else None)

    # ax.scatter(*zip(*observations), label=f'{t}', depthshade=False, s=40)

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

    # no = 1

    ax.scatter(a, c, no, depthshade=False, s=100, color=color, marker=marker, label=f'Test' if i == 0 else None)

ax.set_xlabel('area')
ax.set_ylabel('compactness')
ax.set_zlabel('objects')

plt.legend()
plt.show()
