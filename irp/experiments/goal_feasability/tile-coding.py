from irp.experiments.goal_feasability.env2 import Env2
import irp.experiments.goal_feasability.q2 as q2
import irp.utils
import irp.envs
import irp
import numpy as np
from irp.wrappers import Discretize, MultiSample
from irp.envs.ultrasound.ultra_sound_env import UltraSoundEnv
import matplotlib.pyplot as plt
import cv2
import os
from scipy.ndimage import median_filter
from gym.wrappers import TimeLimit

params = {
    'episodes': 5000, 'alpha': 0.3, 'gamma': 0.6,
    'epsilon': 1, 'epsilon_decay': 0.01, 'min_eps': 0.05, 'learn_delay': 0
}

# Define configuration for creating initial subimages
train_name, s_width, s_height, overlap = 'case10_10.png', 16, 8, 0.75
shape = (512, 512)

# Get all the subimages
train = irp.utils.make_sample_label(train_name, idx=None, width=s_width, height=s_height, overlap=overlap)[0]

coord = (272, 176)
n_size, delta, n_thresholds = 2, 0.08, 6

# Create data specifically for training on the current coordinate
train_Xy = irp.utils.get_neighborhood_images(
    train[0], train[1], coord, shape, s_width, s_height, overlap, n_size
)
train_Xy = zip(*train_Xy)

# Filter so that we only have images that are actually solvable
train_Xy = [Xy for Xy in train_Xy if irp.utils.get_best_dissimilarity(*Xy, n_thresholds)[0] > delta]
X, y = train_Xy[0]

env = Env2(X, y, n_thresholds, delta=0.15)

qtable = {}

for ti in env.intensities:
    bitmask = irp.envs.utils.apply_threshold(X, ti)

    qtable[str(bitmask.flatten().tolist())] = [0., 0., 0.]

    continue
    plt.title(str(irp.envs.utils.compute_dissimilarity(bitmask, y)))
    plt.imshow(np.hstack([bitmask, y]), vmin=0, vmax=1, cmap='gray')
    plt.show()

episodes, alpha, gamma = params['episodes'], params['alpha'], params['gamma']
learn_delay, epsilon_decay, min_eps = params['learn_delay'], params['epsilon_decay'], params['min_eps']
epsilon = params['epsilon']

for e in range(episodes):
    state = env.reset(threshold_i=n_thresholds)
    done = False

    # Until the agent gets stuck in a hole or reaches the goal, keep training it
    while not done:
        rnd = np.random.random()

        if rnd < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(qtable[state])
            
        # Implement this action and move the agent in the desired direction
        new_state, reward, done, info = env.step(action)

        # Update Q(s,a)
        qtable[state][action] = qtable[state][action] + \
                                alpha * (reward + gamma * np.max(qtable[new_state]) - qtable[state][action])
        
        # Update our current state
        state = new_state

    if e >= learn_delay:
        epsilon = max(epsilon - epsilon_decay, min_eps)

[print(key[0:8], value) for key, value in qtable.items()]

# # env = TimeLimit(env, 50)
# qtable = q2.learn(env, **params, write_log=False)

# for key, value in qtable.items():
#     bitmask = np.asarray(eval(key)).reshape((s_width, s_height))

#     plt.title(str(value))
#     plt.imshow(bitmask, cmap='gray', vmin=0, vmax=1)
#     plt.show()

# print(qtable)

s = env.reset(threshold_i=0)

for i in range(10):
    a = np.argmax(qtable[s]); d, i = env.step(a)[-2:]

    print(d, i)