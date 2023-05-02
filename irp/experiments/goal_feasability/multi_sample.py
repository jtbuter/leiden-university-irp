from irp.experiments.goal_feasability.env import Env
import irp.experiments.goal_feasability.q as q
import irp.utils
import irp.envs
import numpy as np
from irp.wrappers import Discretize, MultiSample
from irp.envs.ultrasound.ultra_sound_env import UltraSoundEnv
import matplotlib.pyplot as plt
import cv2
from gym.wrappers import TimeLimit

width, height = 16, 8
subimages, sublabels = np.asarray(irp.utils.get_subimages(f'case10_10.png', width=width, height=height))

n_thresholds = 6

# subimages = subimages[np.array([717, 718, 749, 750])]; sublabels = sublabels[np.array([717, 718, 749, 750])] # Opgelost?
# subimages = subimages[np.array([912, 913, 944, 945])]; sublabels = sublabels[np.array([912, 913, 944, 945])] # Opgelost?
# subimages = subimages[np.array([720, 721, 752, 753])]; sublabels = sublabels[np.array([720, 721, 752, 753])] # Lastig!
subimages = subimages[np.array([1070, 1071, 1102, 1103])]
sublabels = sublabels[np.array([1070, 1071, 1102, 1103])]

bins = (4, 4, 1)

envs = [
    # TimeLimit(Discretize(Env(sample, label, n_thresholds), [0, 0, 0], [1, 1, bins[2] - 1], bins), 30)
    Discretize(Env(sample, label, n_thresholds), [0, 0, 0], [1, 1, bins[2] - 1], bins)
    for sample, label in zip(subimages, sublabels)
]

env: Env = MultiSample(envs)

# Hyperparameters
params = {
    'episodes': 2000, 'alpha': 0.2, 'gamma': 0.99,
    'epsilon': 1.0, 'epsilon_decay': 0.0025, 'min_eps': 0.05, 'learn_delay': 500
}

qtable = q.learn(
    env, **params
)

subimages, sublabels = np.asarray(irp.utils.get_subimages(f'case10_11.png', width=width, height=height))

# env = Discretize(Env(subimages[717], sublabels[717], n_thresholds), [0, 0, 0], [1, 1, bins[2] - 1], bins)
# env = Discretize(Env(subimages[944], sublabels[944], n_thresholds), [0, 0, 0], [1, 1, bins[2] - 1], bins)
env = Discretize(Env(subimages[1102], sublabels[1102], n_thresholds), [0, 0, 1], [1, 1, bins[2]], bins)

s = env.reset(threshold_i=0)

print(qtable[tuple(s)], env.action_map[np.argmax(qtable[tuple(s)])])

# env.render()

for i in range(15):
    a = np.argmax(qtable[tuple(s)])
    s, r, d, i = env.step(a)

    print(d, i)

# print(qtable)

print(qtable[tuple(s)], env.action_map[np.argmax(qtable[tuple(s)])])
    # env.render()

