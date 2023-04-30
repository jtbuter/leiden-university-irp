from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from irp.experiments.goal_feasability.env import Env
import irp.experiments.goal_feasability.q as q
import irp.utils
import irp.envs
import numpy as np
from irp.wrappers import Discretize
from irp.envs.ultrasound.ultra_sound_env import UltraSoundEnv
import matplotlib.pyplot as plt
import cv2

width, height = 8, 8
idx = 1442

subimages, sublabels = np.asarray(irp.utils.get_subimages('case10_10.png', width=width, height=height))
# subimage, sublabel = subimages[183:186], sublabels[183:186]
subimage_ = subimages[
    np.array([1440, 1441, 1442, 1443, 1444, 1445, 1506])
]
sublabel_ = sublabels[
    np.array([1440, 1441, 1442, 1443, 1444, 1445, 1506])
]

# subimage_, sublabel_ = [subimages[idx]], [sublabels[idx]]

test_subimages, test_sublabels = np.asarray(irp.utils.get_subimages('case10_11.png', width=width, height=height))
test_subimage, test_sublabel = test_subimages[idx], test_sublabels[idx]

n_thresholds = 15

best_overlap = -1
best_dims = None

for i in range(1000):
    train_states, test_states = [], []
    train_bit_masks, test_bit_masks = [], []
    dims = tuple(np.random.choice(range(1, 140), 2, replace=True))
    grid = Discretize.make_state_bins(dims=dims, lows=[0, 0], highs=[1, 1])

    for threshold_i in range(15):
        for size in [0]:
            for subimage, sublabel in zip(subimage_, sublabel_):
                intensities = np.linspace(np.min(subimage), np.max(subimage), n_thresholds, dtype=np.uint8).tolist()
                intensity = intensities[threshold_i]
                bit_mask = irp.envs.utils.apply_threshold(subimage, intensity)
                bit_mask = irp.envs.utils.apply_opening(bit_mask, size)
                cont_state = UltraSoundEnv.observation(bit_mask)
                cont_state = cont_state[:2]
                train_bit_masks.append(str(cont_state))
                disc_state = irp.utils.discrete(cont_state, grid)

                train_states.append(disc_state)

            intensities = np.linspace(np.min(test_subimage), np.max(test_subimage), n_thresholds, dtype=np.uint8).tolist()
            intensity = intensities[threshold_i]
            bit_mask = irp.envs.utils.apply_threshold(test_subimage, intensity)
            bit_mask = irp.envs.utils.apply_opening(bit_mask, size)
            cont_state = UltraSoundEnv.observation(bit_mask)
            cont_state = cont_state[:2]
            test_bit_masks.append(str(cont_state))
            disc_state = irp.utils.discrete(cont_state, grid)

            test_states.append(disc_state)

    train_states = set(train_states)
    test_states = set(test_states)
    train_bit_masks = set(train_bit_masks)
    test_bit_masks = set(test_bit_masks)
    overlap = len(train_states & test_states)

    # print(list(train_bit_masks)[0])

    if overlap > best_overlap:
        print(round(len(train_states) / len(train_bit_masks), 3), round(len(test_states) / len(test_bit_masks), 3), len(train_states), len(test_states), dims, overlap)
        # print('unique bitmasks train test', len(train_bit_masks), len(test_bit_masks))

        best_overlap = overlap
        best_dims = dims

print(best_overlap, best_dims)