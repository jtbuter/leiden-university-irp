import matplotlib.pyplot as plt
import numpy as np
import os

from gym.wrappers import TimeLimit
from stable_baselines3.common.callbacks import CallbackList

from irp.callbacks import EvalCallback, LogDissimilarityCallback
from irp.envs.multi_sample import MultiSampleEnv
from irp import utils
from irp import envs, ROOT_DIR
from irp.q import Q
from irp.wrappers import Discretize

subimages, sublabels = utils.get_subimages('case10_11.png')
subimages_, sublabels_ = subimages[183:186], sublabels[183:186]

subimages, sublabels = utils.get_subimages('case10_10.png')

env = MultiSampleEnv([subimages_[0]], [sublabels_[0]], 15)
env = Discretize(env, lows=[0, 0, 0], highs=[1, 1, 4], bins=[5, 3, 5])
env = TimeLimit(env, 100)

eval_env = MultiSampleEnv([subimages[184]], [sublabels[184]], 15)
eval_env = Discretize(eval_env, lows=[0, 0, 0], highs=[1, 1, 4], bins=[5, 3, 5])
# eval_env = TimeLimit(eval_env, 50)

model = Q(env,
    learning_rate=1, tensorboard_log=os.path.join(ROOT_DIR, 'results/multi_sample'), gamma=0.6,
    exploration_delay=0.0, exploration_fraction=0.1
)

model.learn(
    10000,
    callback=CallbackList([
        LogDissimilarityCallback(log_freq=500),
        # EvalCallback(eval_env)
    ]))

print(model.policy.q_table)