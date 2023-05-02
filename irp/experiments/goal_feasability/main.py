from irp.experiments.goal_feasability.env import Env
import irp.experiments.goal_feasability.q as q
import irp.utils
import irp.envs
import numpy as np
from irp.wrappers import Discretize, MultiSample
from irp.envs.ultrasound.ultra_sound_env import UltraSoundEnv
import matplotlib.pyplot as plt
import cv2
from gym.wrappers import TimeLimit

width, height = 16, 8
subimages, sublabels = np.asarray(irp.utils.get_subimages(f'case10_10.png', width=width, height=height))

n_thresholds = 6

subimages = subimages[np.array([1070, 1071, 1102, 1103])]
sublabels = sublabels[np.array([1070, 1071, 1102, 1103])]

dims = (4, 4, 2)

for i, subimage in zip([1070, 1071, 1102, 1103], subimages):
    areas, compactnesses, objects = [], [], []
    train_pairs = []
    tis = np.linspace(np.min(subimage), np.max(subimage), n_thresholds)

    for ti in tis:
        bitmask = irp.envs.utils.apply_threshold(subimage, ti)
        area, compactness, obj = np.round(np.asarray(UltraSoundEnv.observation(bitmask)), 3)

        train_pairs.append((area, compactness, obj))

        areas.append(area)
        compactnesses.append(compactness)
        objects.append(obj)

    print(i, 'odd one' if i == 1103 else '')
    print('train area', sorted(set(areas)))
    print('train compactness', sorted(set(compactnesses)))
    print('train objects', sorted(set(objects)))
    print('train pairs', set(train_pairs))
    print('train disc pairs', set(irp.utils.discrete(pair, Discretize.make_state_bins(dims, (0, 0, 1), (1, 1, dims[2]))) for pair in train_pairs))

    print()

    

# raise Exception

# print('train area', sorted(set(areas)))
# print('train compactness', sorted(set(compactnesses)))
# print('train objects', sorted(set(objects)))

subimages, sublabels = np.asarray(irp.utils.get_subimages(f'case10_11.png', width=width, height=height))

subimages = subimages[np.array([1102])]
sublabels = sublabels[np.array([1102])]

areas, compactnesses, objects = [], [], []

test_pairs = []

for subimage in subimages:
    tis = np.linspace(np.min(subimage), np.max(subimage), n_thresholds)

    for ti in tis:
        bitmask = irp.envs.utils.apply_threshold(subimage, ti)
        area, compactness, obj = np.round(np.asarray(UltraSoundEnv.observation(bitmask)), 3)

        test_pairs.append((area, compactness, obj))

        areas.append(area)
        compactnesses.append(compactness)
        objects.append(obj)

print('test area', sorted(set(areas)))
print('test compactness', sorted(set(compactnesses)))
print('test objects', sorted(set(objects)))
print('test pairs', set(test_pairs))
print('test disc pairs', set(irp.utils.discrete(pair, Discretize.make_state_bins(dims, (0, 0, 1), (1, 1, dims[2]))) for pair in test_pairs))

dims = (4, 4, 2)

train_discs = set()

# for pair in train_pairs:
    # train_discs.add(irp.utils.discrete(pair, Discretize.make_state_bins(dims, (0, 0, 1), (1, 1, dims[2]))))

# for pair in test_pairs:
#     print(irp.utils.discrete(pair, Discretize.make_state_bins(dims, (0, 0, 1), (1, 1, dims[2]))) in train_discs)


# print(train_discs)

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import irp.utils

# a = np.zeros((7, 7), dtype=np.uint8)
# a[5:6,4:6] = 255

# cnts = irp.utils.get_contours(a)
# biggest = max(cnts, key=cv2.contourArea)

# print(cv2.arcLength(biggest, True), irp.utils.get_compactness(biggest, 2))

# plt.imshow(a); plt.show()
