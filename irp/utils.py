from typing import Callable, List, Optional, Tuple, Union
import cv2, os
import itertools
import gym
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

import irp
import irp.envs as envs

from irp.envs.sahba_env import Env

def read_image(
    path: str
) -> np.ndarray:
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def read_sample(
    filename: str,
    preprocess: Optional[bool] = True,
    median_size: Optional[int] = 7
) -> Tuple[np.ndarray, np.ndarray]:
    base_path = os.path.join(irp.ROOT_DIR, "../../data/trus/")
    image_path = os.path.join(base_path, 'images', filename)
    label_path = os.path.join(base_path, 'labels', filename)

    sample, label = read_image(image_path), read_image(label_path)

    if preprocess:
        sample = scipy.ndimage.median_filter(sample, median_size)

    return sample, label

def stacked_read_sample(
    *filenames: List[str],
    preprocess: bool = True,
    median_size: int = 7
) -> List[Tuple[np.ndarray, np.ndarray]]:
    samples = []

    for filename in filenames:
        samples.append(read_sample(filename, preprocess, median_size))

    return samples

def extract_coordinates(
    shape: Tuple[int, int],
    subimage_width: int,
    subimage_height: int,
    overlap: Optional[float] = 0
) -> List[Tuple[int, int]]:
    width, height = shape
    height_step_size = int(subimage_height * (1 - overlap))
    width_step_size = int(subimage_width * (1 - overlap))

    coords = []

    for y in range(0, height - (subimage_height - height_step_size), height_step_size):
        for x in range(0, width - (subimage_width - width_step_size), width_step_size):
            coords.append((x, y))
 
    return coords

# def evaluate(environment: gym.Env, policy, max_steps: Optional[int] = 10, wait_for_done: Optional[bool] = False) -> Tuple[float, bool, np.ndarray]:
#     opt_states = []

#     for ti in range(len(environment.intensity_spectrum)):
#         for vj in range(len(environment.openings)):
#             state, info = environment.reset(ti=ti, vi=vj)

#             if info['d_sim'] == environment.d_sim_opt:
#                 opt_states.append(tuple(state))

#     state, info = tuple(environment.reset(ti=0, vi=0))

#     states = []
#     configs = []
#     values = []

#     for i in range(max_steps):
#         max_q_val = max(policy.values(state)[environment.action_mask()])
#         config = environment.configuration

#         states.append(tuple(state))
#         values.append(max_q_val)
#         configs.append(config)

#         cycle = find_repeating_path(configs)

#         if cycle:
#             states = np.asarray(states)
#             values = np.asarray(values)
#             configs = np.asarray(configs)

#             predicted_state = tuple(states[cycle[0] + np.argmin(values[cycle])])
#             found = predicted_state in opt_states

#             return found

#         action = policy.predict(state, environment.action_mask)
#         state, reward, done, info = environment.step(action)

#     return False

def evaluate(
    environment: gym.Env,
    policy,
    steps: int = 10,
    ti: Optional[Union[int, Tuple]] = None,
    detect_oscillation: Optional[bool] = True
) -> Tuple[np.ndarray, float]:
    # Keep state-value-similarity-config history
    histories = []

    state, info = environment.reset(ti=ti)
    
    histories.append((np.max(policy.values(state)), info['d_sim'], info['configuration'], environment.bitmask))

    for i in range(steps):
        action = policy.predict(state, environment.action_mask, deterministic=True)
        state, reward, done, info = environment.step(action)
    
        histories.append((np.max(policy.values(state)), info['d_sim'], info['configuration'], environment.bitmask))

    values, similarities, configs, bitmasks = [np.asarray(history) for history in zip(*histories)]

    # If we don't want to return the terminal state based on the oscillation property, just return the last found bitmask
    if not detect_oscillation:
        return bitmasks[-1], similarities[-1]

    path = find_repeating_path(list(map(tuple, configs)))
    
    # Otherwise, check if there was a repeating path
    if path:
        best_bitmask = bitmasks[np.argmin(values[path]) + path[0]]
        best_d_sim = similarities[np.argmin(values[path]) + path[0]]

        return best_bitmask, best_d_sim

    # There was no path, so find the best guess of the optimal dissimilarity
    return bitmasks[np.argmin(values)], similarities[np.argmin(values)]

def extract_subimages(
    *samples: np.ndarray,
    subimage_width: int,
    subimage_height: int,
    overlap: Optional[float] = 0,
    return_coords: Optional[bool] = False
) -> List[Union[Tuple[np.ndarray, List], np.ndarray]]:
    height, width = samples[0].shape
    results = []

    if return_coords:
        coords = extract_coordinates((width, height), subimage_width, subimage_height, overlap)

    height_step_size = int(round(subimage_height * (1 - overlap)))
    width_step_size = int(round(subimage_width * (1 - overlap)))

    for sample in samples:
        subimages = []
        sizes = []

        i = 0

        for y in range(0, height - (subimage_height - height_step_size), height_step_size):
            for x in range(0, width - (subimage_width - width_step_size), width_step_size):
                subimage = sample[y:y + subimage_height, x:x + subimage_width]

                subimages.append(subimage)
                sizes.append(subimage.size)

                i += 1

        assert len(set(sizes)) == 1, f"Subimages have differing sizes: {set(sizes)}"

        if return_coords:
            results.append((np.asarray(subimages), coords))
        else:
            results.append(np.asarray(subimages))

    return results

def id_to_coord(
    id: int,
    shape: Tuple[int, int],
    subimage_width: int,
    subimage_height: int,
    overlap: Optional[float] = 0
) -> Tuple[int, int]:
    height, width = shape
    i = 0

    height_step_size = int(round(subimage_height * (1 - overlap), 0))
    width_step_size = int(round(subimage_width * (1 - overlap), 0))

    for y in range(0, height - (subimage_height - height_step_size), height_step_size):
        for x in range(0, width - (subimage_width - width_step_size), width_step_size):
            if i == id:
                return (x, y)

            i += 1

def coord_to_id(
    coord: Tuple[int, int],
    shape: Tuple[int, int],
    subimage_width: int,
    subimage_height: int,
    overlap: Optional[float] = 0
) -> int:
    height, width = shape
    i = 0

    if isinstance(overlap, float) or isinstance(overlap, int):
        width_step_size = int(round((1 - overlap) * subimage_width, 0))
        height_step_size = int(round((1 - overlap) * subimage_height, 0))
    elif isinstance(overlap, tuple):
        width_step_size, height_step_size = overlap

    for y in range(0, height - (subimage_height - height_step_size), height_step_size):
        for x in range(0, width - (subimage_width - width_step_size), width_step_size):
            if (x, y) == coord:
                return i

            i += 1

def diamond(n):
    a = np.arange(n)
    b = np.minimum(a, a[::-1])
    
    return (b[:, None] + b) >= (n - 1) // 2

def get_neighborhood(
    coord: Union[int, Tuple],
    shape: Tuple[int, int],
    subimage_width: int,
    subimage_height: int,
    overlap: Optional[Union[float, Tuple[int, int]]] = 0,
    n_size: Optional[int] = 1,
    neighborhood = 'moore'
) -> List[Tuple]:
    if isinstance(coord, int): coord = id_to_coord(coord, shape, subimage_width, subimage_height, overlap)
    if isinstance(overlap, float):
        width_step_size = int(round((1 - overlap) * subimage_width, 0))
        height_step_size = int(round((1 - overlap) * subimage_height, 0))
    elif isinstance(overlap, tuple):
        width_step_size, height_step_size = overlap

    x, y = coord
    coords = []

    if isinstance(neighborhood, np.ndarray):
        neighbor_map = neighborhood.flatten()
    elif neighborhood == 'neumann':
        neighbor_map = diamond(n_size * 2 + 1).flatten()
    elif neighborhood == 'moore':
        neighbor_map = np.ones((n_size * 2 + 1, n_size * 2 + 1), dtype=bool).flatten()

    for y_i in range(-n_size, n_size + 1):
        y_i *= height_step_size

        for x_i in range(-n_size, n_size + 1):
            x_i *= width_step_size

            coords.append((x + x_i, y + y_i))

    return list(map(tuple, np.asarray(coords)[neighbor_map]))

def get_neighborhood_images(
    subimages: List[np.ndarray],
    sublabels: List[np.ndarray],
    coord: Union[int, Tuple],
    subimage_width: int,
    subimage_height: int,
    overlap: Optional[float] = 0,
    n_size: Optional[int] = 0,
    shape: Optional[Tuple[int, int]] = (512, 512),
    neighborhood: Optional[Union[str, np.ndarray]] = 'moore'
) -> Tuple[np.ndarray, np.ndarray]:
    neighborhood_coords = get_neighborhood(coord, shape, subimage_width, subimage_height, overlap, n_size, neighborhood)

    neighborhood_ids = [
        coord_to_id(coord_, shape, subimage_width, subimage_height, overlap) for coord_ in neighborhood_coords
    ]

    return np.asarray([subimages[i] for i in neighborhood_ids]), np.asarray([sublabels[i] for i in neighborhood_ids])

def apply_action_sequence(subimage, sequence: List[Union[int, float, str]], fns: List[Callable]) -> np.ndarray:
    fn, action = fns[0], sequence[0]

    if not isinstance(action, tuple):
        action = (action,)

    bitmask = fn(subimage, *action)

    for i in range(1, len(fns)):
        fn, action = fns[i], sequence[i] 

        if not isinstance(action, tuple):
            action = [action]

        bitmask = fn(bitmask, *action)

    return bitmask

def get_best_dissimilarity(
    subimage,
    sublabel,
    actions: List[Union[int, str]],
    fns: List[Callable],
    return_seq = False
) -> Union[float, Tuple[float, List]]:
    best_dissim = np.inf
    best_sequence = None

    for sequence in itertools.product(*actions):
        bitmask = apply_action_sequence(subimage, sequence, fns)
        dissim = envs.utils.compute_dissimilarity(sublabel, bitmask)

        if dissim < best_dissim:
            best_dissim = dissim
            best_sequence = sequence

    if return_seq:
        return float(best_dissim), best_sequence

    return float(best_dissim)

def get_best_dissimilarities(
    subimage,
    sublabel,
    actions: List[Union[int, str]],
    fns: List[Callable],
    return_seq = False
) -> Union[float, Tuple[float, List]]:
    sequences = []
    all_dissims = []
    best_dissim = np.inf

    for sequence in itertools.product(*actions):
        bitmask = apply_action_sequence(subimage, sequence, fns)
        dissim = envs.utils.compute_dissimilarity(sublabel, bitmask)

        if dissim < best_dissim:
            best_dissim = dissim
            
        sequences.append(sequence)
        all_dissims.append(dissim)

    all_dissims = np.asarray(all_dissims)

    if return_seq:
        return float(best_dissim), [sequences[i] for i in np.where(all_dissims == best_dissim)[0]]

    return float(best_dissim)

def show(*sample):
    sample = np.hstack(sample)

    plt.imshow(sample, vmin=0, vmax=255, cmap='gray')
    plt.show()

def area_of_overlap(
    label: np.ndarray,
    bitmask: np.ndarray
) -> float:
    tp = ((label == 255) & (bitmask == 255)).sum()
    fn = ((bitmask == 0) & (label != bitmask)).sum()

    return tp / (tp + fn + 1e-6)

def precision(
    label: np.ndarray,
    bitmask: np.ndarray
) -> float:
    tp = ((label == 255) & (bitmask == 255)).sum()
    fp = ((bitmask == 255) & (label != bitmask)).sum()

    return tp / (tp + fp + 1e-6)

def jaccard(
    label: np.ndarray,
    bitmask: np.ndarray
) -> float:
    if not np.any(label):
        label = np.ones_like(label) * 255
        bitmask = np.logical_not(bitmask) * 255
        
    intersection = np.logical_and(label, bitmask).sum()
    union = np.logical_or(label, bitmask).sum()

    return intersection / (union + 1e-6)

def dice(
    label: np.ndarray,
    bitmask: np.ndarray
) -> float:
    if not np.any(label):
        label = np.ones_like(label) * 255
        bitmask = np.logical_not(bitmask) * 255

    tp = ((label == 255) & (bitmask == 255)).sum()
    fp = ((bitmask == 255) & (label != bitmask)).sum()
    fn = ((bitmask == 0) & (label != bitmask)).sum()

    return (2 * tp) / ((2 * tp + fp + fn) + 1e-6)

def non_decreasing(sequence: List) -> bool:
    return all( x <= y for x, y in zip(sequence, sequence[1:]))

def non_increasing(sequence: List) -> bool:
    return all(x >= y for x, y in zip(sequence, sequence[1:]))

def strictly_increasing(sequence: List):
    return all(x < y for x, y in zip(sequence, sequence[1:]))

def strictly_decreasing(sequence: List):
    return all(x > y for x, y in zip(sequence, sequence[1:]))

def is_oscilating(sequence: List) -> bool:
    return not (strictly_increasing(sequence) or strictly_decreasing(sequence))

def normalize_coord(main_coord: Tuple[int, int], neighbor_coord: Tuple[int, int], x_step: int, y_step: int):
    return neighbor_coord[0] - main_coord[0] + x_step, neighbor_coord[1] - main_coord[1] + y_step

def find_repeating_path(sequence):
    seen = []
    i = 0

    while i < len(sequence):
        item = sequence[i]

        if item not in seen:
            seen.append(item)
        else:
            return list(range(seen.index(item), i + 1)) # TODO: Checken of `i + 1` veranderd kan in `i`

        i += 1

    return []

def simplify_sequence(states, values):
    simple_sequence = []
    i = 0

    while i < len(states):
        # The previous state is the same as the current one
        if not (i > 0 and tuple(states[i - 1]) == tuple(states[i])):
            simple_sequence.append(values[i])

        i += 1

    return simple_sequence

def construct_environment(sample: np.ndarray, label: np.ndarray, wrappers: List, parameters: List):
    environment = Env(sample, label, **parameters[0])

    for wrapper, params in zip(wrappers, parameters[1:]):
        environment = wrapper(environment, **params)

    return environment

def cantor_pairing(a: int, b: int) -> int:
    return int((a + b) * (a + b + 1) / 2 + a)

def indexof(sequence, value):
    return next((idx for idx, val in enumerate(sequence) if val == value), -1)