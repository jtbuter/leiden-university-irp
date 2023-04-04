from wrapper import ExpandDimsWrapper
from env import UltraSoundEnv, PaperUltraSoundEnv, Discretize
from gym.wrappers import TimeLimit
from Q import Q
from callback import CustomCallback
import utils
import numpy as np
import gym

image = utils.read_image("/home/joel/Documents/leiden/introductory_research_project/data/trus/images/case10_11.png")
label = utils.read_image("/home/joel/Documents/leiden/introductory_research_project/data/trus/labels/case10_11.png")
subimages, coords = utils.extract_subimages(image, 64, 64)
sublabels, coords = utils.extract_subimages(label, 64, 64)

subimage = subimages[36]
sublabel = sublabels[36]

height, width = sublabel.shape
lows = {'area': 0., 'compactness': 0., 'objects': 0.}
highs = {'area': 1., 'compactness': 1., 'objects': np.ceil(width / 2) * np.ceil(height / 2)}
bins = (35, 35, 35)

# env = PaperUltraSoundEnv(subimage, sublabel)
# env = Discretize(env, lows, highs, bins)
env = gym.make('FrozenLake-v1')
env = TimeLimit(env, 150)
env = ExpandDimsWrapper(env)

print(env.reset())

exploration_fraction = 0.1
exploration_rate = 0.05
learning_rate = 0.1
gamma = 0.95
callback = CustomCallback()

model = Q(
    env,
    learning_rate=learning_rate,
    gamma=gamma,
    exploration_fraction=exploration_fraction,
    exploration_final_eps=exploration_rate,
    tensorboard_log="logs"
)

model.learn(100000, log_interval=1, tb_log_name=f"run-{exploration_fraction}-{exploration_fraction}", callback=callback)
