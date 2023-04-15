from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

from stable_baselines3.common.policies import BasePolicy

import irp.utils

import numpy as np

class QPolicy(BasePolicy):
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

        self._build()

    def _build(self):
        dims = irp.utils.get_dims(self.observation_space, self.action_space)

        self.q_table = np.zeros(dims)

    def predict(self, observation: Union[np.ndarray, Dict[str, np.ndarray]]):
        return self._predict(observation=observation)

    def _predict(self, observation: Union[np.ndarray, Dict[str, np.ndarray]]):
        observation = tuple(observation)

        return np.argmax(self.q_table[observation])