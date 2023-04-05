import gym
import numpy as np
import cv2
import matplotlib.pyplot as plt
import utils
from stable_baselines3.common import env_checker
from wrapper import ExpandDimsWrapper, Discretize

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
        env_checker.check_env(trus_env, skip_render_check=False)
        env_checker.check_env(paper_env, skip_render_check=False)
        env_checker.check_env(discrete_env)
