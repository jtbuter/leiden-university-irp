from gym.wrappers import TimeLimit
import numpy as np
import matplotlib.pyplot as plt
import argparse

from irp.wrappers import Discretize
from irp.envs import Paper2008UltraSoundEnv
from irp.callbacks import StopOnDone
from irp.q import Q
from irp import utils

from copy import deepcopy

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-el', '--episode-length', default=50, type=int)
    parser.add_argument('-er', '--exploration-rate', default=0.05, type=float)
    parser.add_argument('-g', '--gamma', default=0.95, type=float)
    parser.add_argument('-t', '--num-timesteps', default=200000, type=int)

    return parser.parse_args()

argv = get_args()

data = utils.make_sample_label("case10_11.png", "case10_10.png")
train_image, train_label = data[0]
test_image, test_label = data[1]

height, width = train_label.shape

# Confirmed parameters based on paper or experiments
num_thresholds = 15
exploration_fraction = 0.0
vjs = (0, 2, 5)
lows = {'area': 0., 'compactness': 0., 'objects': 0.}
learning_rate = 0.8

# Unconfirmed parameters
highs = {'area': 1., 'compactness': 1., 'objects': np.ceil(width / 2) * np.ceil(height / 2)}
bins = (35, 35, 35)

episode_length = argv.episode_length
exploration_rate = argv.exploration_rate
gamma = argv.gamma

num_timesteps = argv.num_timesteps
callback = StopOnDone()

# Initialize the environment
env = Paper2008UltraSoundEnv(train_image, train_label, num_thresholds=num_thresholds, vjs=vjs)
env = Discretize(env, lows, highs, bins)
env = TimeLimit(env, episode_length)

model = Q(
    env,
    learning_rate=learning_rate,
    gamma=gamma,
    exploration_fraction=exploration_fraction,
    exploration_final_eps=exploration_rate,
    tensorboard_log="experiments/sahba_2008"
)

model_name = f'g={gamma},el={episode_length},exr={exploration_rate},ts={num_timesteps}'

model.learn(num_timesteps, log_interval=1, tb_log_name=model_name, callback=callback)

q_table_old = deepcopy(model.policy.q_table)

model.save(f'experiments/sahba_2008/models/{model_name}')

model = Q.load(path=f'experiments/sahba_2008/models/{model_name}', env=env)

q_table_new = deepcopy(model.policy.q_table)

print(q_table_old.shape, q_table_new.shape)

assert np.all(q_table_new == q_table_old), "Not equal tables"

# Copy parameter list so we don't mutate the original dict
data = model.__dict__.copy()

# Exclude is union of specified parameters (if any) and standard exclusions
included = set([]).union(model._included_save_params())
keys = list(data.keys())

# Remove the parameters entries to be excluded
for param_name in keys:
    if param_name not in included:
        data.pop(param_name, None)

print(data)

# env = Paper2008UltraSoundEnv(train_image, train_label, num_thresholds=num_thresholds, vjs=vjs)
# env = Discretize(env, lows, highs, bins)
# env = TimeLimit(env, episode_length)

# current_state = env.reset()

# plt.title(str(env.threshold_ids) + ' ' + str(env.vj))
# env.render()

# done = False

# while not False:
#     action = model.predict(current_state, deterministic=True)

#     next_state, reward, done, info = env.step(action)
#     current_state = next_state

#     plt.title(str(env.threshold_ids) + ' ' + str(env.vj))
#     env.render()