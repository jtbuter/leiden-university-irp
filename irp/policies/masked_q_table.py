from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

from irp.policies import QPolicy
from irp import utils

import numpy as np
import cv2

SelfMaskedQPolicy = TypeVar("SelfMaskedQPolicy", bound="MaskedQPolicy")

class MaskedQPolicy(QPolicy):
    def predict(
        self: SelfMaskedQPolicy,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        action_mask: np.ndarray
    ):
        return self._predict(observation=observation, action_mask=action_mask)

    def _predict(
        self: SelfMaskedQPolicy,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        action_mask: np.ndarray
    ):
        observation = tuple(observation)
        values = self.q_table[observation]

        return self._masked_argmax(values, action_mask)

    def _masked_argmax(self, values, action_mask):
        return np.argmax(values)

        return max(0, cv2.minMaxLoc(values, mask=action_mask)[3][1])