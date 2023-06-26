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
    action_mapping = list(itertools.product([-1, 1, 0], [-1, 1, 0])); action_mapping.remove((0, 0))

    def __init__(self, sample: np.ndarray, label: np.ndarray, n_thresholds: int, opening: Optional[int] = 0):
        self.sample = sample
        self.label = label

        self.action_space = gym.spaces.Discrete(n=len(self.action_mapping))

        self._intensity_spectrum = envs.utils.get_intensity_spectrum(sample, n_thresholds, add_minus=True)
        
        self.n_thresholds = len(self._intensity_spectrum)
        self.opening = opening
        self.n_features = None

        self._d_sim_opt = irp.utils.get_best_dissimilarity(
            sample, label,
            [itertools.product(self._intensity_spectrum,self._intensity_spectrum), [opening]],
            [envs.utils.apply_threshold, envs.utils.apply_opening]
        )

    def step(self, action: int) -> Tuple[Tuple[float, float, int], int, bool, Dict[str, float]]:
        # Observe the transition after this action, and update our local parameters
        self.ti_left, self.ti_right, self.bitmask = self.transition(action)

        # Compute the dissimilarity metric
        d_sim = envs.utils.compute_dissimilarity(self.label, self.bitmask)

        # Did we improve the dissimilarity compared to the previous timestep
        if d_sim < self._d_sim_old:
            reward = 1
        else:
            reward = -1

        # Update the dissimilarity for the next timestep
        self._d_sim_old = d_sim

        # We're done if we match the best dissimilarity
        done = d_sim <= self._d_sim_opt

        return UltraSoundEnv.observation(self.bitmask), reward, done, {'d_sim': d_sim}

    def transition(self, action: int) -> Tuple[int, np.ndarray]:
        # Compute the new threshold index and intensity
        ti_left_u, ti_right_u = self.action_mapping[action]
        ti_left, ti_right = self.ti_left + ti_left_u, self.ti_right + ti_right_u
        th_left, th_right = self._intensity_spectrum[ti_left], self._intensity_spectrum[ti_right]

        # Compute the new bitmask
        bitmask = envs.utils.apply_threshold(self.sample, th_left, th_right)
        bitmask = envs.utils.apply_opening(bitmask, self.opening)

        return ti_left, ti_right, bitmask

    def reset(self, ti: Optional[Tuple[int, int]] = None) -> Tuple[Tuple[float, float, int], Dict[str, float]]:
        # Pick random threshold intensity, or use the one specified by the user
        self.ti_left, self.ti_right = sorted(np.random.randint(0, self.n_thresholds, 2)) if ti is None else ti
        th_left, th_right = self._intensity_spectrum[self.ti_left], self._intensity_spectrum[self.ti_right]

        # Compute the bitmask and compute the dissimilarity metric
        self.bitmask = envs.utils.apply_threshold(self.sample, th_left, th_right)
        self.bitmask = envs.utils.apply_opening(self.bitmask, self.opening)

        self._d_sim_old = envs.utils.compute_dissimilarity(self.label, self.bitmask)

        return UltraSoundEnv.observation(self.bitmask), {'d_sim': self._d_sim_old}

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n)

        for action in range(self.action_space.n):
            ti_left_u, ti_right_u = self.action_mapping[action]
            ti_left, ti_right = self.ti_left + ti_left_u, self.ti_right + ti_right_u

            # Check if this action doesn't make the index go out of bounds
            valid = ti_left >= 0 and ti_right >= 0 and ti_left < self.n_thresholds and ti_right < self.n_thresholds

            # Check if this action doesn't make the left threshold higher than the right threshold
            valid = valid and ti_left <= ti_right

            mask[action] = valid

        return mask.astype(bool)

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
