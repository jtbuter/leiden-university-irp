import cv2
import numpy as np
from irp import utils
from irp.envs import UltraSoundEnv

train_path = 'case10_10.png'
test_path = 'case10_11.png'

(train_x, train_y), (test_x, test_y) = utils.make_sample_label(train_path, test_path)

lows = [0, 0]
highs = [1, 1]

keys = range(len(lows))
bins = dict(zip(keys, (140, 140)))
state_bins = np.asarray([
    np.linspace(lows[key], highs[key], bins[key] + 1)[1:-1] for key in keys
])

train_tis = np.linspace(np.min(train_x), np.max(train_x), 15)
test_tis = np.linspace(np.min(test_x), np.max(test_x), 15)

train_bit_masks = []
train_states = []
train_bins = []

test_bit_masks = []
test_states = []
test_bins = []

for ti_left in train_tis:
    for ti_right in train_tis:
        bit_mask: np.ndarray = cv2.inRange(train_x, int(ti_left), int(ti_right))
        state = UltraSoundEnv.observation(bit_mask)[:2]
        binned = utils.discrete(state, state_bins)

        train_bit_masks.append(str(bit_mask.flatten()))
        train_states.append(state)
        train_bins.append(binned)

for ti_left in test_tis:
    for ti_right in test_tis:
        bit_mask: np.ndarray = cv2.inRange(test_x, int(ti_left), int(ti_right))
        state = UltraSoundEnv.observation(bit_mask)[:2]
        binned = utils.discrete(state, state_bins)

        test_bit_masks.append(str(bit_mask.flatten()))
        test_states.append(state)
        test_bins.append(binned)

print("unique masks", len(set(train_bit_masks)), len(set(test_bit_masks)))
print("unique states", len(set(train_states)), len(set(test_states)))
print("unique bins", len(set(train_bins)), len(set(test_bins)))

print("overlapping masks", len(set(train_bit_masks) & set(test_bit_masks)))
print("overlapping states", len(set(train_states) & set(test_states)))
print("overlapping bins", len(set(train_bins) & set(test_bins)))