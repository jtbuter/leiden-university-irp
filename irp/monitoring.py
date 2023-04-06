import numpy as np, gym
from gym.wrappers import TimeLimit
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from env import UltraSoundEnv
import utils
import matplotlib.pyplot as plt
import cv2

image = utils._read_image("/home/joel/Documents/leiden/introductory_research_project/data/trus/images/case10_11.png")
label = utils._read_image("/home/joel/Documents/leiden/introductory_research_project/data/trus/labels/case10_11.png")
subimages, coords = utils._extract_subimages(image, 64, 64)
sublabels, coords = utils._extract_subimages(label, 64, 64)

subimage = subimages[36]
sublabel = sublabels[36]

env = UltraSoundEnv(subimage, sublabel)
# env = TimeLimit(env, 150)
model = DQN("MlpPolicy", env, tensorboard_log="./tensor-logs", verbose=0)

class CustomCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        print(self.locals['rewards'].item(), self.locals['dones'].item())

model.learn(150, callback=CustomCallback(), log_interval=1)

# class TensorboardCallback(BaseCallback):
#     """
#     Custom callback for plotting additional values in tensorboard.
#     """

#     def __init__(self, verbose = 0):
#         super().__init__(verbose)

#     def _on_step(self) -> bool:
#         # Log scalar value (here a random variable)
#         value = np.random.random()
#         self.logger.record("random_value", value)

#         return True

# model.learn(200000, callback = TensorboardCallback())