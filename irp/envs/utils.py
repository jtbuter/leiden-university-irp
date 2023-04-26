from typing import Optional, List

import cv2
import numpy as np

def update_thresholds(
    action: int,
    action_map: np.ndarray,
    old_tis: np.ndarray,
    n_thresholds: int
) -> np.ndarray:
    return np.clip(old_tis + action_map[action], a_min=0, a_max=n_thresholds - 1)

def apply_threshold(
    sample: np.ndarray,
    ti_left: int,
    ti_right: Optional[int] = None
) -> np.ndarray:
    ti_left = int(ti_left)

    # Only use a single cut-off value
    if ti_right is None:
        bit_mask = cv2.threshold(sample, ti_left, 255, cv2.THRESH_BINARY_INV)[1]
    else:
        ti_right = int(ti_right)
        bit_mask = cv2.inRange(sample, ti_left, ti_right)

    return bit_mask

def apply_opening(
    bit_mask: np.ndarray,
    size: int
) -> np.ndarray:
    # Check that the structuring element has a size
    if size == 0:
        return bit_mask

    # Construct the kernel, and apply an opening to the bit-mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    bit_mask = cv2.morphologyEx(bit_mask, cv2.MORPH_OPEN, kernel)

    return bit_mask

def get_intensity_spectrum(
    sample: np.ndarray,
    n_thresholds: int
) -> List[int]:
    minimum, maximum = np.min(sample), np.max(sample)

    return np.linspace(minimum, maximum, n_thresholds, dtype=np.uint8)

def compute_dissimilarity(
    bit_mask: np.ndarray,
    label: np.ndarray
) -> float:
    # return np.sum(np.logical_xor(bit_mask, label)) / label.size

    height, width = label.shape
    return (np.sum(np.logical_xor(bit_mask, label)) / (height * width)).astype(float)
