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

class Env(gym.Env):
    action_map = [-1, 0, 1]

    def __init__(self, sample, label, n_thresholds) -> None:
        super().__init__()

        self.sample = sample
        self.label = label
        self.n_thresholds = n_thresholds
        self.threshold_i = None
        self.intensities = np.linspace(np.min(sample), np.max(sample), n_thresholds, dtype=np.uint8).tolist()

        self.action_space = gym.spaces.Discrete(n=3)

    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self.threshold_i = self.threshold_i + self.action_map[action]
        self.threshold_i = np.clip(self.threshold_i, 0, self.n_thresholds - 1)
        intensity = self.intensities[self.threshold_i]
        bit_mask = irp.envs.utils.apply_threshold(self.sample, intensity)
        d_sim = irp.envs.utils.compute_dissimilarity(bit_mask, self.label)
        done = d_sim < 0.05
        reward = 1 if done else 0
        state = str(bit_mask.flatten().tolist())
        state = int(hashlib.sha256(state.encode('utf-8')).hexdigest(), 16) % 10**8
        
        self.d_sim = d_sim

        return state, reward, done, {'dissim': d_sim}

    def reset(self):
        self.threshold_i = self.n_thresholds - 1
        intensity = self.intensities[self.threshold_i]
        bit_mask = irp.envs.utils.apply_threshold(self.sample, intensity)
        state = str(bit_mask.flatten().tolist())
        state = int(hashlib.sha256(state.encode('utf-8')).hexdigest(), 16) % 10**8

        self.d_sim = irp.envs.utils.compute_dissimilarity(bit_mask, self.label)

        return state

    def render(self, mode):
        intensity = self.intensities[self.threshold_i]
        bit_mask = irp.envs.utils.apply_threshold(self.sample, intensity)

        plt.imshow(np.hstack([bit_mask, self.label]), cmap='gray', vmin=0, vmax=255)
        plt.show()