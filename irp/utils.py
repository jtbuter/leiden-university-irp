from typing import Callable, List, Optional, Tuple, Union
import cv2, os
import itertools
import gym
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

import irp
import irp.envs as envs

def read_image(
    path: str
) -> np.ndarray:
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def read_sample(
    filename: str,
    preprocess: Optional[bool] = True
) -> Tuple[np.ndarray, np.ndarray]:
    base_path = os.path.join(irp.ROOT_DIR, "../../data/trus/")
    image_path = os.path.join(base_path, 'images', filename)
    label_path = os.path.join(base_path, 'labels', filename)

    sample, label = read_image(image_path), read_image(label_path)

    if preprocess:
        sample = scipy.ndimage.median_filter(sample, 7)

    return sample, label

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

def evaluate(environment: gym.Env, policy, max_steps: Optional[int] = 10, wait_for_done: Optional[bool] = False) -> Tuple[float, bool, np.ndarray]:
    done = False
    state, info = environment.reset(ti=0)
    tis = [environment.ti]
    bitmask = environment.bitmask.copy()
    is_done = False

    for t in range(max_steps):
        action = policy.predict(state, lambda: environment.action_mask())
        state, reward, done, info = environment.step(action)

        tis.append(environment.ti)

        if is_oscilating(tis) is True:
            break
        elif tis[-1] == tis[-2]:
            is_done = done

            break

        bitmask = environment.bitmask.copy()
        is_done = done

    return info['d_sim'], is_done, bitmask

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

    if isinstance(overlap, float):
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

    if (tp + fn) == 0.0:
        return 1.0

    return tp / (tp + fn)

def precision(
    label: np.ndarray,
    bitmask: np.ndarray
) -> float:
    tp = ((label == 255) & (bitmask == 255)).sum()
    fp = ((bitmask == 255) & (label != bitmask)).sum()

    if (tp + fp) == 0.0:
        return 1.0

    return tp / (tp + fp)

def jaccard(
    label: np.ndarray,
    bitmask: np.ndarray
) -> float:
    intersection = np.logical_and(label, bitmask).sum()
    union = np.logical_or(label, bitmask).sum()

    if union == 0.0:
        return 1.0

    return intersection / union

def dice(
    label: np.ndarray,
    bitmask: np.ndarray
) -> float:
    tp = ((label == 255) & (bitmask == 255)).sum()
    fp = ((bitmask == 255) & (label != bitmask)).sum()
    fn = ((bitmask == 0) & (label != bitmask)).sum()

    if (tp + fp + fn) == 0:
        return 1.0

    return (2 * tp) / (2 * tp + fp + fn)

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
