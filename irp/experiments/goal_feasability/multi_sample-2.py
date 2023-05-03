from irp.experiments.goal_feasability.env import Env
import irp.experiments.goal_feasability.q as q
import irp.utils
import irp.envs
import numpy as np
from irp.wrappers import Discretize, MultiSample
from irp.envs.ultrasound.ultra_sound_env import UltraSoundEnv
import matplotlib.pyplot as plt
import cv2
import os
from scipy.ndimage import median_filter
from gym.wrappers import TimeLimit


subimage_width, subimage_height = 16, 8
filename = 'case10_11.png'
n_thresholds = 6
bins = (5, 4, 2)
# Hyperparameters
params = {
    'episodes': 5000, 'alpha': 0.3, 'gamma': 0.9,
    'epsilon': 1.0, 'epsilon_decay': 0.0025, 'min_eps': 0.05, 'learn_delay': 1000
}
overlap = 0.5
n_size = 1

# Define the paths to the related parent directories
base_path = os.path.join(irp.GIT_DIR, "../data/trus/")
image_path = os.path.join(base_path, 'images')
label_path = os.path.join(base_path, 'labels')

# Read the image and label
image = irp.utils.read_image(os.path.join(image_path, filename))
label = irp.utils.read_image(os.path.join(label_path, filename))

image = median_filter(image, 7)

subimages = np.asarray(irp.utils.extract_subimages(image, subimage_width, subimage_height, overlap=overlap)[0])
sublabels = np.asarray(irp.utils.extract_subimages(label, subimage_width, subimage_height, overlap=overlap)[0])

# Create train images
neighborhood = irp.utils.get_neighborhood((240, 272), image, subimage_width, subimage_height, overlap=overlap, n_size=n_size)

subimages_ = []
sublabels_ = []

for neighbor in neighborhood:
    id = irp.utils.coord_to_id(neighbor, (512, 512), 16, 8, overlap=overlap)
    subimage = subimages[id]
    sublabel = sublabels[id]

    mini, maxi = np.min(subimage), np.max(subimage)

    tis = np.linspace(mini, maxi, n_thresholds, dtype=np.uint8).tolist()
    tis = np.concatenate(([mini - 1], tis))

    best_dissim = np.inf
    best_bitmask = -1

    for ti in tis:
        bitmask = irp.envs.utils.apply_threshold(subimage, ti)
        dissim = irp.envs.utils.compute_dissimilarity(bitmask, sublabel)

        if dissim < best_dissim:
            best_dissim = dissim
            best_bitmask = bitmask

    if best_dissim <= 0.08:
        subimages_.append(subimage)
        sublabels_.append(sublabel)

envs = [
    TimeLimit(Discretize(Env(sample, label, n_thresholds), [0, 0, 1], [1, 1, bins[2]], bins), 30)
    for sample, label in zip(subimages_, sublabels_)
]

env: Env = MultiSample(envs)


qtable = q.learn(
    env, **params
)

print(qtable)

filename = 'case10_10.png'

# Define the paths to the related parent directories
base_path = os.path.join(irp.GIT_DIR, "../data/trus/")
image_path = os.path.join(base_path, 'images')
label_path = os.path.join(base_path, 'labels')

# Read the image and label
image = irp.utils.read_image(os.path.join(image_path, filename))
label = irp.utils.read_image(os.path.join(label_path, filename))

image = median_filter(image, 7)

# Create test images
subimages = np.asarray(irp.utils.extract_subimages(image, subimage_width, subimage_height, overlap=overlap)[0])
sublabels = np.asarray(irp.utils.extract_subimages(label, subimage_width, subimage_height, overlap=overlap)[0])
id = irp.utils.coord_to_id((240, 272), (512, 512), subimage_width, subimage_height, overlap=overlap)

# env = Discretize(Env(subimages[717], sublabels[717], n_thresholds), [0, 0, 0], [1, 1, bins[2] - 1], bins)
# env = Discretize(Env(subimages[944], sublabels[944], n_thresholds), [0, 0, 0], [1, 1, bins[2] - 1], bins)
env = Discretize(Env(subimages[id], sublabels[id], n_thresholds), [0, 0, 1], [1, 1, bins[2]], bins)

s = env.reset(threshold_i=0)

print(qtable[tuple(s)], env.action_map[np.argmax(qtable[tuple(s)])])

# env.render()

for i in range(15):
    a = np.argmax(qtable[tuple(s)])
    s, r, d, i = env.step(a)

    print(d, i)

print(qtable[tuple(s)], env.action_map[np.argmax(qtable[tuple(s)])])
    # env.render()

