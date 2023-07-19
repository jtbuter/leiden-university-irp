from typing import Dict, List, Optional, Tuple

import itertools
import numpy as np

import irp.utils
import irp.envs.env as env
import irp.envs as envs

from irp.envs.base_env import UltraSoundEnv

class MorphedEnv(env.Env):
    action_mapping = list(itertools.product([-1, 1, 0], [-1, 1, 0]))

    def __init__(self, sample: np.ndarray, label: np.ndarray, n_thresholds: int, opening: Optional[List[int]] = [], sahba: Optional[bool] = True):
        super().__init__(sample, label, n_thresholds, [0], sahba)
        
        self._openings = opening
        self.n_openings = len(self._openings)

        self._d_sim_opt = irp.utils.get_best_dissimilarity(
            sample, label,
            [self._intensity_spectrum, self._openings],
            [envs.utils.apply_threshold, envs.utils.apply_opening]
        )

    def step(self, action: int) -> Tuple[Tuple[float, float, int], int, bool, Dict[str, float]]:
        # Observe the transition after this action, and update our local parameters
        self.ti_left, self.vi, self.bitmask = self.transition(action)

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
            'configuration': (self.ti_left, self.vi)
        }

        return UltraSoundEnv.observation(self.bitmask), reward, done, info

    def transition(self, action: int) -> Tuple[int, np.ndarray]:
        # Compute the new threshold index and intensity
        ti_left_u, vi_u = self.action_mapping[action]
        ti_left, vi = self.ti_left + ti_left_u, self.vi + vi_u

        ti_left = min(max(0, ti_left), self.n_thresholds - 1)
        vi = min(max(0, vi), self.n_openings - 1)

        th_left, vj = self._intensity_spectrum[ti_left], self._openings[vi]

        # Compute the new bitmask
        bitmask = envs.utils.apply_threshold(self.sample, th_left)
        bitmask = envs.utils.apply_opening(bitmask, vj)

        return ti_left, vi, bitmask

    def reset(self, ti: Optional[Tuple[int, int]] = None) -> Tuple[Tuple[float, float, int], Dict[str, float]]:
        # Pick random threshold intensities
        self.ti_left = np.random.randint(0, self.n_thresholds)
        self.vi = np.random.randint(0, self.n_openings)

        # For now use ti both for specifying ti_left and vi
        if ti is not None:
            self.ti_left, self.vi = ti

        th_left, vj = self._intensity_spectrum[self.ti_left], self._openings[self.vi]

        # Compute the bitmask and compute the dissimilarity metric
        self.bitmask = envs.utils.apply_threshold(self.sample, th_left)
        self.bitmask = envs.utils.apply_opening(self.bitmask, vj)

        d_sim = envs.utils.compute_dissimilarity(self.label, self.bitmask)

        self._d_sim_old = d_sim

        info = {
            'd_sim': d_sim,
            'configuration': (self.ti_left, self.vi)
        }

        return UltraSoundEnv.observation(self.bitmask), info

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n)

        for action in range(self.action_space.n):
            ti_left_u, vi_u = self.action_mapping[action]
            ti_left, vi = self.ti_left + ti_left_u, self.vi + vi_u

            # Check if this action doesn't make the index go out of bounds
            valid = ti_left >= 0 and vi >= 0 and ti_left < self.n_thresholds and vi < self.n_openings

            mask[action] = valid

        return mask.astype(bool)

    # TODO: Remove this placeholder
    def action_mask(self) -> np.ndarray:
        return np.ones(self.action_space.n, dtype=bool)
