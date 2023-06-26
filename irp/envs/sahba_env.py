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
    threshold_actions = [-1, 1]
    morphological_actions = [0, 2, 5]

    action_mapping = list(itertools.product(threshold_actions, morphological_actions))

    def __init__(self, sample: np.ndarray, label: np.ndarray, n_thresholds: int, opening: Optional[int] = 0):
        self.sample = sample
        self.label = label

        self.action_space = gym.spaces.Discrete(n=len(self.action_mapping))
        self._intensity_spectrum = envs.utils.get_intensity_spectrum(sample, n_thresholds, add_minus=True)
        
        self.n_thresholds = len(self._intensity_spectrum)

        self._d_sim_opt = irp.utils.get_best_dissimilarity(
            sample, label, [self._intensity_spectrum, self.morphological_actions], [envs.utils.apply_threshold, envs.utils.apply_opening]
        )

    def step(self, action: int) -> Tuple[Tuple[float, float, int], int, bool, Dict[str, float]]:
        # Observe the transition after this action, and update our local parameters
        self.ti, self.vj, self.bitmask = self.transition(action)

        # Compute the dissimilarity metric
        d_sim = envs.utils.compute_dissimilarity(self.label, self.bitmask)

        # Did we improve the dissimilarity compared to the previous timestep
        if d_sim < self._d_sim_old:
            reward = 10
        else:
            reward = 0

        # Update the dissimilarity for the next timestep
        self._d_sim_old = d_sim

        # We're done if we match the best dissimilarity
        done = d_sim <= self._d_sim_opt

        return UltraSoundEnv.observation(self.bitmask), reward, done, {'d_sim': d_sim}

    def transition(self, action: int) -> Tuple[int, int, np.ndarray]:
        ti_u, vj = self.action_mapping[action]
        ti = self.ti + ti_u
        th = self._intensity_spectrum[ti]

        # Compute the new bitmask
        bitmask = envs.utils.apply_threshold(self.sample, th)
        bitmask = envs.utils.apply_opening(bitmask, vj)

        return ti, vj, bitmask

    def reset(self, ti: Optional[int] = None, vj: Optional[int] = None) -> Tuple[Tuple[float, float, int], Dict[str, float]]:
        # Pick random threshold intensity and kernel size, or use the ones specified by the user
        self.ti = np.random.randint(0, self.n_thresholds) if ti is None else ti
        self.vj = np.random.choice(self.morphological_actions).item() if vj is None else vj

        # Retrieve the intensity
        th = self._intensity_spectrum[self.ti]

        # Construct the bitmask and compute the dissimilarity metric
        self.bitmask = envs.utils.apply_threshold(self.sample, th)
        self.bitmask = envs.utils.apply_opening(self.bitmask, self.vj)
        self._d_sim_old = envs.utils.compute_dissimilarity(self.label, self.bitmask)

        return UltraSoundEnv.observation(self.bitmask), {'d_sim': self._d_sim_old}

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n)

        for action in range(self.action_space.n):
            # We only require the update to the index
            ti_u = self.action_mapping[action][0]
            ti = self.ti + ti_u

            # Check if this action doesn't make the index go out of bounds
            mask[action] = ti >= 0 and ti < self.n_thresholds

        return mask.astype(bool)

    def guidance_mask(self) -> np.ndarray:
        mask = np.ones(self.action_space.n)

        return mask.astype(bool)

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
