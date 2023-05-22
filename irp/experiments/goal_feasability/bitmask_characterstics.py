from matplotlib import pyplot as plt
import numpy as np

import irp
import irp.utils
import irp.envs as envs
import irp.wrappers as wrappers
from irp.envs.ultrasound.ultra_sound_env import UltraSoundEnv

id_to_loc_map = [
    'bottom-left', 'bottom-center', 'bottom-right',
    'center-left', 'middle', 'center-right',
    'top-left', 'top-center', 'top-right',
]

# Train and test filename, and characteristics of the subimages
train_name = 'case10_10.png'
test_name = 'case10_11.png'
subimage_width, subimage_height = 16, 8
overlap = 0.75
n_size = 0

# Get all the subimages
train_Xs, train_ys = np.asarray(
    irp.utils.make_sample_label(train_name, width=subimage_width, height=subimage_height, overlap=overlap, idx=None)
)[0]
test_Xs, test_ys = np.asarray(
    irp.utils.make_sample_label(test_name, width=subimage_width, height=subimage_height, overlap=0, idx=None)
)[0]

# coordinates = [(256, 176), coord, (240, 184), (224, 192)]
# ids = np.asarray([
#     irp.utils.coord_to_id(coord, (512, 512), subimage_width, subimage_height, overlap) for coord in coordinates
# ])
# 
# train_Xs = train_Xs[ids]
# train_ys = train_ys[ids]

coord = (272, 176)

idx = irp.utils.coord_to_id(coord, (512, 512), subimage_width, subimage_height, overlap)
train_Xs, train_ys = irp.utils.get_neighborhood_images(
    train_Xs, train_ys, idx, subimage_width, subimage_height, overlap, n_size=n_size, neighborhood='neumann'
)

print(len(train_Xs))

# irp.utils.show(train_ys[4])

n_thresholds = 4

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection='3d')

for t, (sample, label) in enumerate(zip(train_Xs, train_ys)):
    print(t)
    observations = []

    for i, ti in enumerate(envs.utils.get_intensity_spectrum(sample, n_thresholds, add_minus=False)[:-1]):
        bitmask = envs.utils.apply_threshold(sample, ti)
        # bitmask = envs.utils.apply_opening(bitmask, 1)
        a, c, no = UltraSoundEnv.observation(bitmask)

        print(a, c, no)

        observations.append((a, c, no))

    ax.scatter(*zip(*observations), label=f'{t}', depthshade=False, s=40)

idx = irp.utils.coord_to_id(coord, (512, 512), subimage_width, subimage_height, 0)
sample = test_Xs[idx]

observations = []

print(t+1)

for i, ti in enumerate(envs.utils.get_intensity_spectrum(sample, n_thresholds, add_minus=False)[:-1]):
    bitmask = envs.utils.apply_threshold(sample, ti)
    # bitmask = envs.utils.apply_opening(bitmask, 1)
    a, c, no = UltraSoundEnv.observation(bitmask)

    print(a, c, no)

    observations.append((a, c, no))

ax.scatter(*zip(*observations), label='Test', depthshade=False, s=80)

ax.set_xlabel('area')
ax.set_ylabel('compactness')
ax.set_zlabel('objects')

plt.legend()
plt.show()
