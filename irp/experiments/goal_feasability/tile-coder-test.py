import numpy as np

import irp.wrappers as wrappers
import matplotlib.pyplot as plt
import irp
import irp.utils
import irp.wrappers as wrappers
import irp.experiments.tile_coding.q as q
import irp.experiments.tile_coding.env as env
from irp.experiments.tile_coding.policy import TiledQTable

# Hyperparameters
parameters = {
    'learning_delay': 0,  # Delay until epsilon starts updating
    'episodes': 5000,       # Total number of episodes
    'alpha': 0.6,           # Learning rate
    'gamma': 0.95,           # Discount factor
    'epsilon': 1.0,         # Amount of randomness in the action selection
    'epsilon_decay': 0.001, # Fixed amount to decrease
    'n_thresholds': 2,
    'min_epsilon': 0.05
}

# Define some tiling parameters
tiles_per_dim = (4, 4, 4)
value_limits = [(0, 1), (0, 1), (0, 32)]
tilings = 32
alpha = parameters['alpha']

# Define bitmask characteristics
train_name = 'case10_10.png'
subimage_width, subimage_height = 16, 8
n_size, overlap = 1, 0.875
n_thresholds = 5

# Get the specific label we're looking for
coord = (288, 216)
idx = irp.utils.coord_to_id(coord, (512, 512), subimage_width, subimage_height, overlap)

# Get all the training subimages
Xs, ys = np.asarray(
    irp.utils.make_sample_label('case10_10.png', width=subimage_width, height=subimage_height, overlap=overlap, idx=None)
)[0]

# Extract the neighborhood of the images we were looking for
train_Xs, train_ys = irp.utils.get_neighborhood_images(
    Xs, ys, coord, subimage_width, subimage_height, overlap=overlap, n_size=n_size
)

environments = wrappers.MultiSample()

for sample, label in zip(train_Xs, train_ys):
    environment = env.Env(sample, label, n_thresholds)
    environment = wrappers.Tiled(environment, tiles_per_dim, tilings, value_limits)
    # environment = wrappers.Goal(environment) # TODO: Moeten we een wrapper gebruiken, of kunnen we vermenigvuldigen met `done`

    environments.add(environment)

q.learn(environments, parameters, log=True)