from typing import Dict, Optional, Tuple

import itertools
import numpy as np

import irp.utils
import irp.envs.env as env
import irp.envs as envs

from irp.envs.base_env import UltraSoundEnv

class RangedEnv(env.Env):
    action_mapping = list(itertools.product([-1, 1, 0], [-1, 1, 0]))

    def __init__(self, sample: np.ndarray, label: np.ndarray, n_thresholds: int, opening: Optional[int] = 0, sahba: Optional[bool] = True):
        super().__init__(sample, label, n_thresholds, opening, sahba)

        self._threshold_indices = self.make_threshold_indices(n_thresholds)
        self._d_sim_opt = irp.utils.get_best_dissimilarity(
            sample, label,
            [itertools.product(self._intensity_spectrum, self._intensity_spectrum), [opening]],
            [envs.utils.apply_threshold, envs.utils.apply_opening]
        )

    def step(self, action: int) -> Tuple[Tuple[float, float, int], int, bool, Dict[str, float]]:
        # Observe the transition after this action, and update our local parameters
        self.ti_left, self.ti_right, self.bitmask = self.transition(action)

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
            'configuration': (self.ti_left,)
        }

        return UltraSoundEnv.observation(self.bitmask), reward, done, info

    def transition(self, action: int) -> Tuple[int, np.ndarray]:
        # Compute the new threshold index and intensity
        ti_left_u, ti_right_u = self.action_mapping[action]
        ti_left, ti_right = self.ti_left + ti_left_u, self.ti_right + ti_right_u

        ti_left = min(max(0, ti_left), self.n_thresholds - 1)
        ti_right = min(max(0, ti_right), self.n_thresholds - 1)

        th_left, th_right = self._intensity_spectrum[ti_left], self._intensity_spectrum[ti_right]

        # Compute the new bitmask
        bitmask = envs.utils.apply_threshold(self.sample, th_left, th_right)
        bitmask = envs.utils.apply_opening(bitmask, self.opening)

        return ti_left, ti_right, bitmask

    def reset(self, ti: Optional[Tuple[int, int]] = None) -> Tuple[Tuple[float, float, int], Dict[str, float]]:
        # Select random index for initial threshold indices
        ti_index = np.random.randint(len(self._threshold_indices))

        # Pick random threshold intensity, or use the one specified by the user
        self.ti_left, self.ti_right = self._threshold_indices[ti_index] if ti is None else ti
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

            # Check if this action doesn't make the left threshold higher than the right threshold (but allow 0, 0)
            valid = valid and (ti_left < ti_right or (ti_left == 0 and ti_right == 0))

            mask[action] = valid

        return mask.astype(bool)

    @classmethod
    def make_threshold_indices(self, n_thresholds):
        combinations = itertools.product(range(n_thresholds), range(n_thresholds))
        combinations = list(filter(lambda x: x[0] < x[1], combinations))

        combinations.append((0, 0))

        return combinations