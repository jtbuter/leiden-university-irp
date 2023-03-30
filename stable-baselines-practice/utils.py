import numpy as np

def __process_thresholds(action, action_map, tis, n_thresholds):
    return np.clip(tis + action_map[action], 0, n_thresholds - 1)


def __compute_dissimilarity(bit_mask, label):
    height, width = label.shape

    return np.sum(np.logical_xor(bit_mask, label)) / (height * width)
