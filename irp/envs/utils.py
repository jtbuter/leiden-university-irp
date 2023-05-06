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
        bitmask = cv2.threshold(sample, ti_left, 255, cv2.THRESH_BINARY_INV)[1]
    else:
        ti_right = int(ti_right)
        bitmask = cv2.inRange(sample, ti_left, ti_right)

    return bitmask

def apply_opening(
    bitmask: np.ndarray,
    size: int
) -> np.ndarray:
    # Check that the structuring element has a size
    if size == 0:
        return bitmask

    # Construct the kernel, and apply an opening to the bit-mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    bitmask = cv2.morphologyEx(bitmask, cv2.MORPH_OPEN, kernel)

    return bitmask

def get_intensity_spectrum(
    sample: np.ndarray,
    n_thresholds: int
) -> List[int]:
    minimum, maximum = np.min(sample), np.max(sample)

    return np.linspace(minimum, maximum, n_thresholds, dtype=np.uint8)

def compute_dissimilarity(
    bitmask: np.ndarray,
    label: np.ndarray
) -> float:
    return (bitmask != label).sum() / label.size

def get_contours(
    bitmask: np.ndarray
) -> np.ndarray:
    return cv2.findContours(bitmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

def get_compactness(
    contour: np.ndarray,
    object_area: float
) -> float:
    perimeter = cv2.arcLength(contour, True)

    return (4 * np.pi * object_area) / (perimeter ** 2)

def get_area(
    contour: np.ndarray
) -> float:
    return cv2.contourArea(contour)

def normalize_area(
    bitmask: np.ndarray,
    object_area: float
) -> float:
    return object_area / bitmask.size