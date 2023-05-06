from typing import Optional
import gym
import gym.spaces
import numpy as np
import irp.envs as envs

class Env(gym.Env):
    # Defines how thresholds can be modified
    action_mapping = [-1, 0, 1]

    def __init__(self, sample: np.ndarray, label: np.ndarray, n_thresholds: int):
        self.sample = sample
        self.label = label
        self.n_thresholds = n_thresholds

        self.action_space = gym.spaces.Discrete(n=len(self.action_mapping))

        self._intensity_spectrum = envs.utils.get_intensity_spectrum(sample, n_thresholds)

    def step(self, action: int):
        # Update the threshold index
        self.ti = min(max(0, self.ti + self.action_mapping[action]), self.n_thresholds - 1)
        th = self._intensity_spectrum[self.ti]

        # Compute the bitmask
        bitmask = envs.utils.apply_threshold(self.sample, th)

        # Update the dissimilarity metric
        d_sim = envs.utils.compute_dissimilarity_new(bitmask, self.label)

        done = d_sim < 0.15 # We're done

        # Did we make an improvement or not
        if done:
            reward = 10
        else:
            reward = 0
        
        self._d_sim = d_sim

        return str(bitmask.flatten().tolist()), reward, done, {'d_sim': d_sim}

    def reset(self, ti: Optional[int] = None):
        # Pick random threshold intensity, or use the one specified by the user
        self.ti = np.random.randint(0, self.n_thresholds) if ti is None else ti
        th = self._intensity_spectrum[self.ti]

        # Compute the bitmask
        bitmask = envs.utils.apply_threshold(self.sample, th)

        # Update the dissimilarity metric
        self._d_sim = envs.utils.compute_dissimilarity_new(bitmask, self.label)

        return str(bitmask.flatten().tolist())
