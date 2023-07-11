from typing import Dict, List, Optional, Tuple, Union

import itertools
import gym
import gym.spaces
import numpy as np
import random

import irp.envs as envs
import irp.utils
from irp.envs.base_env import UltraSoundEnv

class Env(gym.Env):
    # Define how parameters can be modified, and remove the neutral action
    action_mapping = [-1, 1, 0]
    
    # The number of features the state-vector consists of
    n_features = 3

    def __init__(self, sample: np.ndarray, label: np.ndarray, n_thresholds: int, opening: Optional[int] = 0):
        self.sample = sample
        self.label = label

        self.action_space = gym.spaces.Discrete(n=len(self.action_mapping))

        self._intensity_spectrum = envs.utils.get_intensity_spectrum(sample, n_thresholds, add_minus=True)
        
        self.n_thresholds = len(self._intensity_spectrum)
        self.opening = opening

        self._d_sim_opt = irp.utils.get_best_dissimilarity(
            sample, label,
            [self._intensity_spectrum],
            [envs.utils.apply_threshold]
        )

    def step(self, action: int) -> Tuple[Tuple[float, float, int], int, bool, Dict[str, float]]:
        # Observe the transition after this action, and update our local parameters
        self.ti_left, self.bitmask = self.transition(action)

        # Compute the dissimilarity metric
        d_sim = envs.utils.compute_dissimilarity(self.label, self.bitmask)

        # Did we improve the dissimilarity compared to the previous timestep
        if d_sim <= self._d_sim_opt:
            reward = 1
        else:
            reward = -1

        # We're done if we match the best dissimilarity
        done = d_sim <= self._d_sim_opt

        return UltraSoundEnv.observation(self.bitmask), reward, done, {'d_sim': d_sim}

    def transition(self, action: int) -> Tuple[int, np.ndarray]:
        # Compute the new threshold index and intensity
        ti_left_u = self.action_mapping[action]
        ti_left = self.ti_left + ti_left_u

        ti_left = min(max(0, ti_left), self.n_thresholds - 1)

        th_left = self._intensity_spectrum[ti_left]

        # Compute the new bitmask
        bitmask = envs.utils.apply_threshold(self.sample, th_left)

        return ti_left, bitmask

    def reset(self, ti: Optional[Tuple[int, int]] = None) -> Tuple[Tuple[float, float, int], Dict[str, float]]:
        # Pick random threshold intensity, or use the one specified by the user
        self.ti_left = np.random.randint(self.n_thresholds) if ti is None else ti
        th_left = self._intensity_spectrum[self.ti_left]

        # Compute the bitmask and compute the dissimilarity metric
        self.bitmask = envs.utils.apply_threshold(self.sample, th_left)

        self._d_sim_old = envs.utils.compute_dissimilarity(self.label, self.bitmask)

        return UltraSoundEnv.observation(self.bitmask), {'d_sim': self._d_sim_old}

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n)

        for action in range(self.action_space.n):
            ti_left_u = self.action_mapping[action]
            ti_left = self.ti_left + ti_left_u

            # Check if this action doesn't make the index go out of bounds
            valid = ti_left >= 0 and ti_left < self.n_thresholds

            mask[action] = valid

        return mask.astype(bool)

    # TODO: Remove this placeholder
    def action_mask(self) -> np.ndarray:
        return np.ones(self.action_space.n, dtype=bool)

    def guidance_mask(self) -> np.ndarray:
        action_mask = self.action_mask()
        mask = np.zeros(self.action_space.n)

        for action in range(self.action_space.n):
            # Ensure we don't try to use any invalid actions
            if action_mask[action] == False:
                continue

            # Perform a reversible transition
            _, _, bitmask = self.transition(action)

            # Compute the dissimilarity metric
            d_sim = envs.utils.compute_dissimilarity(self.label, bitmask)

            # Check if this action would give a positive reward
            mask[action] = d_sim < self._d_sim_old

        # If no actions return a positive reward, allow all actions to be chosen
        if not np.any(mask):
            mask = np.ones(self.action_space.n)

        return mask.astype(bool)

    # TODO: Remove this placeholder
    def guidance_mask(self) -> np.ndarray:
        return np.ones(self.action_space.n, dtype=bool)

    def _randint(self, start: int, stop: int, exclude: Optional[List] = []):
        include = range(start, stop)

        return random.choice([ti for ti in include if ti not in exclude])

    @property
    def d_sim_opt(self) -> float:
        return self._d_sim_opt

    @property
    def d_sim_old(self) -> float:
        return self._d_sim_old

    @property
    def d_sim(self) -> float:
        return self._d_sim

    @property
    def intensity_spectrum(self) -> np.ndarray:
        return self._intensity_spectrum
