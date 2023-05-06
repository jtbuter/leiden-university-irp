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

# Train filename and characteristics of the subimages
train_name = 'case10_10.png'
subimage_width, subimage_height = 16, 8
overlap = 0.75
shape = (512, 512)

# Get all the subimages
subimages, sublabels = np.asarray(
    irp.utils.make_sample_label(train_name, width=subimage_width, height=subimage_height, overlap=overlap, idx=None)
)[0]

# Define environment settings
n_thresholds = 6
bins = (3, 3, 4)
n_size = 2
delta = 0.08

# Hyperparameters for learning
params = {
    'episodes': 5000, 'alpha': 0.3, 'gamma': 0.9,
    'epsilon': 1.0, 'epsilon_decay': 0.0025, 'min_eps': 0.05, 'learn_delay': 1000
}
# Coordinate we're analyzing right now
# coord = (256, 184) # TODO: This is a difficult one, learn how to evaluate border subimages
coords = [(272, 176), (256, 184), (288, 184), (304, 192), (224, 200), (320, 200), (320, 208), (208, 216), (320, 216), (208, 224), (320, 224), (208, 232), (336, 232), (192, 240), (192, 248), (192, 256), (192, 264), (272, 264), (336, 264), (208, 272), (240, 272), (256, 272), (336, 272), (208, 280), (224, 280), (304, 280), (320, 280), (336, 280), (304, 288), (320, 288)]
coords = [(272, 232)]

for coord in coords:
    idx = irp.utils.coord_to_id(coord, shape, subimage_width, subimage_height, overlap)

    # Get the subimages in the neighborhood of the image we're analyzing
    n_subimages, n_sublabels = irp.utils.get_neighborhood_images(
        subimages, sublabels, coord, shape, subimage_width, subimage_height, overlap, n_size
    )

    # Prepare images to actually use for training
    train_Xy = []

    # For now, only use subimage that actually contain a goal-state
    for subimage, sublabel in zip(n_subimages, n_sublabels):
        solvable = irp.utils.get_best_dissimilarity(subimage, sublabel, n_thresholds) <= delta

        if solvable:
            train_Xy.append((subimage, sublabel))

    print(f"{coord}: Using {len(train_Xy)} samples to train on" + (', continuing...' if len(train_Xy) == 0 else ''))

    # We don't actually have any images to train on
    if len(train_Xy) == 0:
        continue

    # Create a MultiSample environment to train on multiple subimages
    envs = MultiSample([])

    for subimage, sublabel in train_Xy:
        env = Env(subimage, sublabel, n_thresholds, delta)
        env = Discretize(env, [0, 0, 1], [1, 1, bins[2]], bins)

        envs.add(env)

    # Train the model
    qtable = q.learn(envs, **params, write_log=True)

    # Define the test filename and get all the subimages
    test_name = 'case10_11.png'
    subimage, sublabel = np.asarray(
        irp.utils.make_sample_label(test_name, width=subimage_width, height=subimage_height, overlap=overlap, idx=idx)
    )[0]

    # Set-up and evaluate the environment
    env = Env(subimage, sublabel, n_thresholds, delta)
    env = Discretize(env, [0, 0, 1], [1, 1, bins[2]], bins)
    s = env.reset(threshold_i=0)

    for i in range(n_thresholds + 1):
        a = np.argmax(qtable[tuple(s)]); d, i = env.step(a)[-2:]

        print(d, i)

    # env.render()
