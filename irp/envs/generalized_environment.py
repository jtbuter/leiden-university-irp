from typing import Dict, List, Optional, Tuple, Union

import itertools
import gym
import gym.spaces
import numpy as np
import random

import irp.envs as envs
import irp.utils
from irp.envs.base_env import UltraSoundEnv

class GeneralizedEnv(gym.Env):
    # Define how parameters can be modified, and remove the neutral action
    action_mapping = list(itertools.product([-1, 1, 0], [-1, 1, 0], [-1, 1, 0]))
    
    # The number of features the state-vector consists of
    n_features = 3

    def __init__(
        self,
        sample: np.ndarray, label: np.ndarray, 
        n_thresholds: int, openings: List[int] = [0],
        sahba: Optional[bool] = True, ranged: Optional[bool] = False
    ):
        self.sample = sample
        self.label = label
        self.sahba = sahba

        self.action_space = gym.spaces.Discrete(n=len(self.action_mapping))

        self._openings = openings
        self._intensity_spectrum = envs.utils.get_intensity_spectrum(sample, n_thresholds, add_minus=True)

        # Generalize single-valued thresholding to ranged thresholding
        generalized_intensities = self.generalize_ranges(ranged, self._intensity_spectrum)

        # Create set of possible intensity indices the agent may use for random initialization
        self.action_dims = np.asarray(tuple(len(i) for i in generalized_intensities) + (len(self._openings),))
        self.threshold_pairs = self.make_threshold_pairs(self.action_dims[:2])

        # Find the optimal dissimilarity
        self._d_sim_opt = irp.utils.get_best_dissimilarity(
            sample, label,
            [itertools.product(*generalized_intensities), self._openings],
            [envs.utils.apply_threshold, envs.utils.apply_opening]
        )

    @classmethod
    def generalize_ranges(self, ranged: bool, intensity_spectrum: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Generalize single-valued to ranged thresholding
        if ranged:
            return (intensity_spectrum, intensity_spectrum)
        else:
            return (np.asarray([-1.0]), intensity_spectrum)

    @classmethod
    def make_threshold_pairs(self, n_thresholds: Tuple[int, int]):
        ti_left_max, ti_right_max = n_thresholds
        index_pairs = []

        for ti_left in range(ti_left_max):
            for ti_right in range(ti_left, ti_right_max):
                index_pairs.append((ti_left, ti_right))

        return index_pairs

    def step(self, action: int) -> Tuple[Tuple[float, float, int], int, bool, Dict[str, float]]:
        # Observe the transition after this action, and update our local parameters
        self.ti_left, self.ti_right, self.vi, self.bitmask = self.transition(action)

        # Compute the dissimilarity metric
        d_sim = envs.utils.compute_dissimilarity(self.label, self.bitmask)

        # Determine the reward based on the achieved dissimilarity
        reward = self.reward(d_sim)

        self._d_sim_old = d_sim

        # We're done if we match the best dissimilarity
        done = d_sim <= self._d_sim_opt

        # Expose the internal configuration and obtained dissimilarity
        info = {
            'd_sim': d_sim,
            'configuration': (self.ti_left, self.ti_right, self.vi)
        }

        return UltraSoundEnv.observation(self.bitmask), reward, done, info

    def reward(self, d_sim: float):
        if self.sahba:
            return 10 if d_sim < self._d_sim_old else 0
        else:
            return 1 if d_sim <= self._d_sim_opt else -1

    def transition(self, action: int) -> Tuple[int, np.ndarray]:
        # Compute the new threshold index and intensity
        ti_left_u, ti_right_u, vi_u = self.action_mapping[action]
        ti_left, ti_right, vi = self.ti_left + ti_left_u, self.ti_right + ti_right_u, self.vi + vi_u

        ti_left, ti_right, vi = np.clip((ti_left, ti_right, vi), (0, 0, 0), self.action_dims - 1).astype(int)

        th_left, th_right = self._intensity_spectrum[ti_left], self._intensity_spectrum[ti_right]
        vj = self._openings[vi]

        # Compute the new bitmask
        bitmask = envs.utils.apply_threshold(self.sample, th_left, th_right)
        bitmask = envs.utils.apply_opening(bitmask, vj)

        return ti_left, ti_right, vi, bitmask

    def reset(self, ti: Optional[Tuple[int, int]] = None) -> Tuple[Tuple[float, float, int], Dict[str, float]]:
        # Pick random threshold intensities and kernel size, or use the ones specified by the user
        if ti is None:
            self.ti_left, self.ti_right = random.choice(self.threshold_pairs)
            self.vi = np.random.randint(0, self.action_dims[2])
        else:
            self.ti_left, self.ti_right, self.vi = ti

        th_left, th_right = self._intensity_spectrum[self.ti_left], self._intensity_spectrum[self.ti_right]
        vj = self._openings[self.vi]

        # Compute the bitmask and compute the dissimilarity metric
        self.bitmask = envs.utils.apply_threshold(self.sample, th_left, th_right)
        self.bitmask = envs.utils.apply_opening(self.bitmask, vj)

        d_sim = envs.utils.compute_dissimilarity(self.label, self.bitmask)

        self._d_sim_old = d_sim

        # Expose the internal configuration and obtained dissimilarity
        info = {
            'd_sim': d_sim,
            'configuration': (self.ti_left, self.ti_right, self.vi)
        }

        return UltraSoundEnv.observation(self.bitmask), info

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n)

        for action in range(self.action_space.n):
            ti_left_u, ti_right_u, vi_u = self.action_mapping[action]
            ti_left, ti_right, vi = self.ti_left + ti_left_u, self.ti_right + ti_right_u, self.vi + vi_u

            ti_left_c, ti_right_c, vi_c = np.clip((ti_left, ti_right, vi), (0, 0, 0), self.action_dims - 1).astype(int)

            # Check if this action doesn't make the index go out of bounds
            valid = ti_left == ti_left_c and ti_right == ti_right_c and vi == vi_c
            
            # Ensure the threshols don't get swapped
            valid = valid and ti_left <= ti_right

            mask[action] = valid

        return mask.astype(bool)

    # TODO: Remove this placeholder
    def action_mask(self) -> np.ndarray:
        return np.ones(self.action_space.n, dtype=bool)

    # def guidance_mask(self) -> np.ndarray:
    #     action_mask = self.action_mask()
    #     mask = np.zeros(self.action_space.n)

    #     for action in range(self.action_space.n):
    #         # Ensure we don't try to use any invalid actions
    #         if action_mask[action] == False:
    #             continue

    #         # Perform a reversible transition
    #         _, _, _, bitmask = self.transition(action)

    #         # Compute the dissimilarity metric
    #         d_sim = envs.utils.compute_dissimilarity(self.label, bitmask)

    #         # Check if this action would give a positive reward
    #         mask[action] = d_sim < self._d_sim_old

    #     # If no actions return a positive reward, allow all actions to be chosen
    #     if not np.any(mask):
    #         mask = np.ones(self.action_space.n)

    #     return mask.astype(bool)

    # TODO: Remove this placeholder
    def guidance_mask(self) -> np.ndarray:
        return np.ones(self.action_space.n, dtype=bool)

    def _randint(self, start: int, stop: int, exclude: Optional[List] = []) -> int:
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
