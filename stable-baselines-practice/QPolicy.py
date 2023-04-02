from typing import Any, Dict, List, Optional, Type, Tuple, Union

import numpy as np
import torch as th
from gym import spaces
from torch import nn

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.dqn.policies import QNetwork
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.type_aliases import Schedule

class QPolicy(BasePolicy):
    """
    Policy class with Q-Value Net and target net for DQN
    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        net_arch: Optional[List[int]] = None,
    ) -> None:
        super().__init__(observation_space, action_space,)

        self.q_table = np.zeros((16, 4))

    def _predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        action = np.argmax(self.q_table[obs])

        return action.reshape(-1)

    def predict(self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        
        return self._predict(observation, deterministic=deterministic), state

MlpPolicy = QPolicy