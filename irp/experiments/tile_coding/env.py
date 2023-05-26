from typing import Optional
import gym
import gym.spaces
import numpy as np
import irp.envs as envs
import irp.utils
from irp.envs.ultrasound.ultra_sound_env import UltraSoundEnv
 
class Env(gym.Env):
    # Defines how thresholds can be modified
    action_mapping = [0, -1, 1]

    def __init__(self, sample: np.ndarray, label: np.ndarray, n_thresholds: int):
        self.sample = sample
        self.label = label

        self.action_space = gym.spaces.Discrete(n=len(self.action_mapping))

        self._intensity_spectrum = envs.utils.get_intensity_spectrum(sample, n_thresholds, add_minus=True)
        
        self.n_thresholds = n_thresholds + 1

        self._d_sim, (self._th,) = irp.utils.get_best_dissimilarity(sample, label, [self._intensity_spectrum], [envs.utils.apply_threshold], return_seq=True)

    def step(self, action: int):
        # Did we reach the terminal state already
        if not np.isinf(self.steps):
            # Update the threshold index
            self.ti = min(max(0, self.ti + self.action_mapping[action]), self.n_thresholds - 1)
            th = self._intensity_spectrum[self.ti]

            # Compute the new bitmask
            self.bitmask = envs.utils.apply_threshold(self.sample, th)

            # Update the dissimilarity metric
            d_sim = envs.utils.compute_dissimilarity(self.bitmask, self.label)

            # Did we reach the best possible dissimilarity
            if d_sim <= self._d_sim:
                reward = 1
            else:
                reward = -1
        else:
            d_sim = self._d_sim
            reward = 0

        # Done if the reward is not negative
        done = reward >= 0
        self.steps += 1

        # Expose the dissimilarity, e.g. for logging
        self.info = {'d_sim': d_sim}

        return UltraSoundEnv.observation(self.bitmask), reward, done, self.info

    def reset(self, ti: Optional[int] = None):
        # Pick random threshold intensity, or use the one specified by the user
        self.ti = np.random.randint(0, self.n_thresholds) if ti is None else ti
        th = self._intensity_spectrum[self.ti]

        # We've generated a terminal state as an initial state
        if th == self._th:
            self.steps = np.inf
        else:
            self.steps = 0
        
        # Compute the bitmask
        self.bitmask = envs.utils.apply_threshold(self.sample, th)

        return UltraSoundEnv.observation(self.bitmask)

    @property
    def d_sim(self) -> float:
        return self._d_sim

    @property
    def intensity_spectrum(self) -> np.ndarray:
        return self._intensity_spectrum