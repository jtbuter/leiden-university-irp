import gym
import numpy as np
import cv2
import matplotlib.pyplot as plt
import utils
from stable_baselines3.common import env_checker
from wrapper import ExpandDimsWrapper

class UltraSoundEnv(gym.Env):
    action_map = np.array([(-1, -1), (-1, 1), (1, -1), (1, 1)])

    def __init__(self, sample = None, label = None, num_thresholds = 10):
        super(UltraSoundEnv, self).__init__()

        self.label = label.copy()
        self.sample = sample.copy()
        self.state = None

        self.action_space = gym.spaces.Discrete(n = 4)
        self.observation_space = gym.spaces.Box(0, 255, self.sample.shape + (1,), np.uint8)

        self.num_thresholds = num_thresholds
        self.threshold_ids = np.array([0, self.num_thresholds - 1])
        self.thresholds = np.linspace(np.min(sample), np.max(sample), num_thresholds, dtype = np.uint8)

        self.old_dissim = None

    def _get_reward(self, dissim):
        if dissim < self.old_dissim:
            return 10
        elif dissim == self.old_dissim:
            return 0
        elif dissim > self.old_dissim:
            return 0

    def step(self, action):
        new_threshold_ids = utils.process_thresholds(action, self.action_map, self.threshold_ids, self.num_thresholds)
        lt, rt = self.thresholds[new_threshold_ids]

        bit_mask = cv2.inRange(self.sample, int(lt), int(rt))
        next_state = cv2.bitwise_and(self.sample, self.sample, mask = bit_mask)

        dissim = utils.compute_dissimilarity(bit_mask, self.label)
        reward = self._get_reward(dissim)
        is_done = bool(dissim < 0.05)

        self.old_dissim = dissim
        self.threshold_ids = new_threshold_ids
        self.state = next_state.reshape(*next_state.shape, 1)

        return self.state, reward, is_done, {}

    def reset(self):
        bit_mask = np.full(self.label.shape, 255)
        self.state = self.sample.reshape(*self.sample.shape, 1)
        self.threshold_ids = np.array([0, self.num_thresholds - 1])
        self.old_dissim = utils.compute_dissimilarity(bit_mask, self.label)

        return self.state

    def render(self):
        plt.imshow(np.squeeze(self.state), cmap = 'gray', vmin = 0, vmax = 1)
        plt.show()

    def close(self):
        pass

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

class Discretize(gym.Env):
    def __init__(self, env, lows, highs, bins):
        self.env = env
        self.observation_space = gym.spaces.MultiDiscrete(bins)
        self.action_space = env.action_space

        self._state_bins = None

        self._setup_env(lows, highs)

    def _setup_env(self, lows, highs):
        keys = lows.keys()
        bins = dict(zip(keys, utils.get_dims(self.observation_space)))

        self._state_bins = np.asarray([
            np.linspace(lows[key], highs[key], bins[key] + 1)[1:-1] for key in keys
        ])

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        return np.asarray(utils.discretize(state, self._state_bins)), reward, done, info

    def reset(self):
        state = self.env.reset()

        return np.asarray(utils.discretize(state, self._state_bins))

if __name__ == "__main__":
    image = utils.read_image("/home/joel/Documents/leiden/introductory_research_project/data/trus/images/case10_11.png")
    label = utils.read_image("/home/joel/Documents/leiden/introductory_research_project/data/trus/labels/case10_11.png")
    subimages, coords = utils.extract_subimages(image, 64, 64)
    sublabels, coords = utils.extract_subimages(label, 64, 64)

    subimage = subimages[36]
    sublabel = sublabels[36]

    trus_env = UltraSoundEnv(subimage, sublabel)
    paper_env = PaperUltraSoundEnv(subimage, sublabel)

    lake_env = gym.make('FrozenLake-v1')
    expand_dims = ExpandDimsWrapper(lake_env)

    assert np.array([lake_env.reset()]).shape == expand_dims.reset().shape

    height, width = sublabel.shape
    lows = {'area': 0., 'compactness': 0., 'objects': 0.}
    highs = {'area': 1., 'compactness': 1., 'objects': np.ceil(width / 2) * np.ceil(height / 2)}
    bins = (35, 35, 35)

    discrete_env = Discretize(paper_env, lows, highs, bins)

    paper_env_state = paper_env.reset()
    paper_env_sample = paper_env.observation_space.sample()

    check_envs = True

    # Check if we're using valid gym environments
    if check_envs:
        env_checker.check_env(trus_env)
        env_checker.check_env(paper_env)
        env_checker.check_env(discrete_env)
