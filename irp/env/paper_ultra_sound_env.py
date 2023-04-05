import gym
import numpy as np
import cv2
import matplotlib.pyplot as plt

from stable_baselines3.common import env_checker

from irp import utils
from irp.env.ultra_sound_env import UltraSoundEnv

class PaperUltraSoundEnv(UltraSoundEnv):
    def __init__(self, sample = None, label = None, num_thresholds = 10):
        super().__init__(sample, label, num_thresholds)

        self.observation_space = gym.spaces.Box(0, np.inf, (3,))        

    def step(self, action):
        # Convert an action to new threshold indices
        new_threshold_ids = utils.process_thresholds(action, self.action_map, self.threshold_ids, self.num_thresholds)

        # Convert indices to gray-values for generalization
        lt, rt = self.thresholds[new_threshold_ids]

        # Extract a bit-mask using the gray-values
        bit_mask = cv2.inRange(self.sample, int(lt), int(rt))

        # Convert the bit-mask to a discrete state
        next_state = self.observation(bit_mask)

        # Compute dissimilarity and convert this to a reward
        dissim = utils.compute_dissimilarity(bit_mask, self.label)
        reward = self._get_reward(dissim)
        is_done = bool(dissim < 0.05)

        self.old_dissim = dissim
        self.threshold_ids = new_threshold_ids
        self.state = next_state

        return np.asarray(self.state, dtype=np.float32), reward, is_done, {}

    def reset(self):
        # Pick two random new threshold indices
        new_threshold_ids = np.random.choice(range(0, self.num_thresholds), 2)

        # Convert indices to gray-values for generalization
        lt, rt = self.thresholds[new_threshold_ids]

        # Extract a bit-mask using the gray-values
        bit_mask = cv2.inRange(self.sample, int(lt), int(rt))

        # Convert the bit-mask to a discrete state
        next_state = self.observation(bit_mask)

        # Compute current dissimilarity
        dissim = utils.compute_dissimilarity(bit_mask, self.label)

        self.old_dissim = dissim
        self.threshold_ids = new_threshold_ids
        self.state = next_state

        return np.asarray(self.state, dtype=np.float32)
        
    def observation(self, bit_mask):
        contours = utils.get_contours(bit_mask)
        num_objects = len(contours)

        if num_objects == 0:
            return (0., 0., 0.)

        # Get the biggest object based on its area
        biggest_object = max(contours, key = cv2.contourArea)
        object_area = utils.get_area(biggest_object)

        if object_area == 0:
            return (0., 0., num_objects)

        compactness = utils.get_compactness(biggest_object, object_area)
        normalized_area = utils.normalize_area(bit_mask, object_area)

        return (normalized_area, compactness, num_objects)