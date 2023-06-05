from typing import List, Optional, Union

import gym
import gym.spaces
import numpy as np
import random

import irp.envs as envs
import irp.utils
from irp.envs.base_env import UltraSoundEnv

class Env(gym.Env):
    # Defines how parameters can be modified
    action_mapping = [-1, 1, 0]

    def __init__(self, sample: np.ndarray, label: np.ndarray, n_thresholds: int):
        self.sample = sample
        self.label = label

        self.action_space = gym.spaces.Discrete(n=len(self.action_mapping))

        self._intensity_spectrum = envs.utils.get_intensity_spectrum(sample, n_thresholds)
        self._intensity_spectrum = np.insert(self._intensity_spectrum, 0, -1.0)
        
        self.n_thresholds = n_thresholds + 1

        d_sim, seq = irp.utils.get_best_dissimilarities(sample, label, [self._intensity_spectrum, [2]], [envs.utils.apply_threshold, envs.utils.apply_opening], return_seq=True)
        seq = np.asarray(seq)[:, 0]

        self._d_sim = d_sim
        self._ti_exc = np.searchsorted(self._intensity_spectrum, seq)

    def step(self, action: int):
        # Update the threshold index
        self.ti = min(max(0, self.ti + self.action_mapping[action]), self.n_thresholds - 1)
        th = self._intensity_spectrum[self.ti]

        # Compute the new bitmask
        self.bitmask = envs.utils.apply_threshold(self.sample, th)
        self.bitmask = envs.utils.apply_opening(self.bitmask, 2)

        # Compute the bitmask and compute the dissimilarity metric
        d_sim = envs.utils.compute_dissimilarity(self.label, self.bitmask)

        # We're done if we match the best dissimilarity
        done = d_sim <= self._d_sim

        # Did we reach the best possible dissimilarity
        if done:
            reward = 1
        else:
            reward = -1

        return UltraSoundEnv.observation(self.bitmask), reward, done, {'d_sim': d_sim}

    def reset(self, ti: Optional[int] = None):
        # Pick random threshold intensity, or use the one specified by the user
        self.ti = np.random.randint(0, self.n_thresholds) if ti is None else ti
        self.ti = self._randint(0, self.n_thresholds, exclude=self._ti_exc) if ti is None else ti
        th = self._intensity_spectrum[self.ti]

        # Compute the bitmask and compute the dissimilarity metric
        self.bitmask = envs.utils.apply_threshold(self.sample, th)
        self.bitmask = envs.utils.apply_opening(self.bitmask, 2)

        d_sim = envs.utils.compute_dissimilarity(self.label, self.bitmask)

        return UltraSoundEnv.observation(self.bitmask), {'d_sim': d_sim}

    def _randint(self, start: int, stop: int, exclude: Optional[List] = []):
        include = range(start, stop)

        return random.choice([ti for ti in include if ti not in exclude])

    @property
    def d_sim(self):
        return self._d_sim

    @property
    def intensity_spectrum(self):
        return self._intensity_spectrum

    @property
    def best_ti(self):
        return self._ti_exc