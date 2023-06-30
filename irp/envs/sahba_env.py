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
    threshold_actions = [-1, 1, 0]
    morphological_actions = [-1, 1, 0]

    action_mapping = list(itertools.product(threshold_actions, morphological_actions)); action_mapping.remove((0, 0))

    def __init__(self, sample: np.ndarray, label: np.ndarray, n_thresholds: int, openings: Optional[List[int]] = [0]):
        self.sample = sample
        self.label = label

        self.action_space = gym.spaces.Discrete(n=len(self.action_mapping))
        self._intensity_spectrum = envs.utils.get_intensity_spectrum(sample, n_thresholds, add_minus=True)
        self._openings = openings

        self.n_thresholds = len(self._intensity_spectrum)
        self.n_openings = len(self._openings)

        self._d_sim_opt = irp.utils.get_best_dissimilarity(
            sample, label, [self._intensity_spectrum, self._openings], [envs.utils.apply_threshold, envs.utils.apply_opening]
        )

    def step(self, action: int) -> Tuple[Tuple[float, float, int], int, bool, Dict[str, float]]:
        # Do one timestep
        self.timesteps += 1

        # Observe the transition after this action, and update our local parameters
        self.ti, self.vi, self.bitmask = self.transition(action)

        # Compute the dissimilarity metric
        d_sim = envs.utils.compute_dissimilarity(self.label, self.bitmask)

        # Did we improve the dissimilarity compared to the previous timestep
        if d_sim <= self._d_sim_opt:
        # if d_sim < self._d_sim_old:
            reward = 1
        else:
            reward = -1

        # Update the dissimilarity for the next timestep
        self._d_sim_old = d_sim

        # We're done if we match the best dissimilarity
        done = d_sim <= self._d_sim_opt

        return UltraSoundEnv.observation(self.bitmask), reward, done, {'d_sim': d_sim}

    def transition(self, action: int) -> Tuple[int, int, np.ndarray]:
        # Compute the new intensity and kernel size indices
        ti_u, vi_u = self.action_mapping[action]
        ti, vi = self.ti + ti_u, self.vi + vi_u

        th = self._intensity_spectrum[ti]
        vj = self._openings[vi]

        # Compute the new bitmask
        bitmask = envs.utils.apply_threshold(self.sample, th)
        bitmask = envs.utils.apply_opening(bitmask, vj)

        return ti, vi, bitmask

    def reset(self, ti: Optional[int] = None, vi: Optional[int] = None) -> Tuple[Tuple[float, float, int], Dict[str, float]]:
        # Reset the number of timesteps
        self.timesteps = 0

        # Pick random threshold intensity and kernel size, or use the ones specified by the user
        self.ti = np.random.randint(0, self.n_thresholds) if ti is None else ti
        self.vi = np.random.randint(0, self.n_openings) if vi is None else vi

        # Retrieve the intensity and opening kernel
        th = self._intensity_spectrum[self.ti]
        vj = self._openings[self.vi]

        # Construct the bitmask and compute the dissimilarity score
        self.bitmask = envs.utils.apply_threshold(self.sample, th)
        self.bitmask = envs.utils.apply_opening(self.bitmask, vj)
        self._d_sim_old = envs.utils.compute_dissimilarity(self.label, self.bitmask)

        return UltraSoundEnv.observation(self.bitmask), {'d_sim': self._d_sim_old}

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n)

        for action in range(self.action_space.n):
            # We only require the update to the index
            ti_u, vi_u = self.action_mapping[action]
            ti, vi = self.ti + ti_u, self.vi + vi_u

            # Check if this action doesn't make the index go out of bounds
            mask[action] = (ti >= 0 and ti < self.n_thresholds) and (vi >= 0 and vi < self.n_openings)

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

    # def guidance_mask(self) -> np.ndarray:
    #     return np.ones(self.action_space.n, dtype=bool)

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

    @property
    def openings(self) -> np.ndarray:
        return self._openings

    @property
    def configuration(self) -> Tuple:
        return self.ti, self.vi