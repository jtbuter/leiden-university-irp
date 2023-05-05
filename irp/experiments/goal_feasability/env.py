import math
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, Tuple
import irp.envs
import irp.utils
import gym
import gym.spaces
import numpy as np
import matplotlib.pyplot as plt
from irp.envs.ultrasound.ultra_sound_env import UltraSoundEnv
import random

class Env(gym.Env):
    action_map = [-1, 0, 1]

    def __init__(self, sample, label, n_thresholds, delta=0.08) -> None:
        super().__init__()

        self.sample = sample
        self.label = label
        self.threshold_i = None
        mini, maxi = np.min(sample), np.max(sample)
        self.intensities = np.linspace(mini, maxi, n_thresholds, dtype=np.uint8).tolist()
        self.intensities = np.concatenate(([mini-1], self.intensities))
        self.n_thresholds = n_thresholds + 1
        self.delta = delta

        self.observation_space = gym.spaces.Discrete(n=self.n_thresholds)
        self.action_space = gym.spaces.Discrete(n=3)

    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self.threshold_i = self.threshold_i + self.action_map[action]
        # self.threshold_i = np.clip(self.threshold_i, 0, self.n_thresholds - 1)
        self.threshold_i = max(0, min(self.n_thresholds - 1, self.threshold_i))
        intensity = self.intensities[self.threshold_i]
        bit_mask = irp.envs.utils.apply_threshold(self.sample, intensity)
        d_sim = irp.envs.utils.compute_dissimilarity(bit_mask, self.label)
        done = d_sim <= self.delta
        reward = 1 if done else 0
        # state = str(bit_mask.flatten().tolist())
        # state = int(hashlib.sha256(state.encode('utf-8')).hexdigest(), 16) % 10**8
        
        state = UltraSoundEnv.observation(bit_mask)

        self.d_sim = d_sim

        return state, reward, done, {'dissim': d_sim}

    def reset(self, threshold_i: int = None):
        # if threshold_i is None:
        #     if np.random.random() < 0.5:
        #         self.threshold_i = self.n_thresholds - 1
        #     else:
        #         self.threshold_i = 0
        if threshold_i is None:
            # self.threshold_i = int(np.random.randint(0, self.n_thresholds))
            # self.threshold_i = math.floor(random.random() * self.n_thresholds)
            self.threshold_i = random.randrange(0, self.n_thresholds)
        else:
            self.threshold_i = threshold_i
            
        intensity = self.intensities[self.threshold_i]
        bit_mask = irp.envs.utils.apply_threshold(self.sample, intensity)
        # state = str(bit_mask.flatten().tolist())
        # state = int(hashlib.sha256(state.encode('utf-8')).hexdigest(), 16) % 10**8

        state = UltraSoundEnv.observation(bit_mask)

        self.d_sim = irp.envs.utils.compute_dissimilarity(bit_mask, self.label)

        return state

    def render(self, mode="human"):
        intensity = self.intensities[self.threshold_i]
        bit_mask = irp.envs.utils.apply_threshold(self.sample, intensity)

        plt.imshow(np.hstack([bit_mask, self.label]), cmap='gray', vmin=0, vmax=255)
        plt.show()
