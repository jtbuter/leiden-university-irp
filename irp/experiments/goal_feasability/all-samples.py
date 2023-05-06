import cProfile
from irp.experiments.goal_feasability.env import Env
import irp.experiments.goal_feasability.q as q
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

# Train and test filename, and characteristics of the subimages
train_name = 'case10_10.png'
test_name = 'case10_11.png'
subimage_width, subimage_height = 16, 8
shape = (512, 512)
overlap = 0.75

# Get all the subimages
train_Xs, train_ys = np.asarray(
    irp.utils.make_sample_label(train_name, width=subimage_width, height=subimage_height, overlap=overlap, idx=None)
)[0]
test_Xs, test_ys = np.asarray(
    irp.utils.make_sample_label(test_name, width=subimage_width, height=subimage_height, overlap=overlap, idx=None)
)[0]

# Define environment settings
n_thresholds = 6
bins = (3, 3, 4)
n_size = 2
delta = 0.08

# Hyperparameters for learning
params = {
    'episodes': 500, 'alpha': 0.8, 'gamma': 0.9,
    'epsilon': 1.0, 'epsilon_decay': 0.0025, 'min_eps': 0.05, 'learn_delay': 1000
}

coords = irp.utils.extract_subimages(np.zeros((512, 512)), subimage_width, subimage_height, 0)[1]

# Store resulting bitmasks to this array to visualize our final results
result = np.zeros((512, 512))
failed = []

for coord in coords:
    x, y = coord

    # Make sure we're not wasting processing power right now
    if not (x >= 192 and x <= 336 and y >= 176 and y <= 288):
        continue

    idx = irp.utils.coord_to_id(coord, shape, subimage_width, subimage_height, overlap)

    # Get the subimages in the neighborhood of the image we're analyzing
    n_subimages, n_sublabels = irp.utils.get_neighborhood_images(
        train_Xs, train_ys, coord, shape, subimage_width, subimage_height, overlap, n_size
    )

    # Prepare images to actually use for training
    train_Xy = []

    # For now, only use subimage that actually contain a goal-state
    for train_X, train_y in zip(n_subimages, n_sublabels):
        solvable = irp.utils.get_best_dissimilarity(train_X, train_y, n_thresholds) <= delta

        if solvable:
            train_Xy.append((train_X, train_y))

    print(f"{coord}: Using {len(train_Xy)} samples to train on" + (', continuing...' if len(train_Xy) == 0 else ''))

    if len(train_Xy) == 0:
        continue

    # Create a MultiSample environment to train on multiple subimages
    envs = MultiSample([])

    for train_X, train_y in train_Xy:
        env = Env(train_X, train_y, n_thresholds, delta)
        env = Discretize(env, [0, 0, 1], [1, 1, bins[2]], bins)

        envs.add(env)

    # Train the model
    qtable = q.learn(envs, **params, write_log=False)

    # Define the test filename and get all the subimages
    test_X, test_y = test_Xs[idx], test_ys[idx]

    # Set-up and evaluate the environment
    env = Env(test_X, test_y, n_thresholds, delta)
    env = Discretize(env, [0, 0, 1], [1, 1, bins[2]], bins)

    # Start at the very lowest threshold, i.e. lower than the lower sample.min() value
    s = env.reset(threshold_i=0)

    # Perform `n_thresholds` timesteps, as we should've found the result by then
    for i in range(10): a = np.argmax(qtable[tuple(s)]); d, i = env.step(a)[-2:]

    print(d, i)

    if not d:
        failed.append(coord)

    intensity = env.intensities[env.threshold_i]
    bit_mask = irp.envs.utils.apply_threshold(test_X, intensity)

    result[y:y + subimage_height, x:x + subimage_width] = bit_mask

print('Failed on', failed)

plt.imshow(result, cmap='gray', vmin=0, vmax=1)
plt.show()