from gym.wrappers.time_limit import TimeLimit
import irp
import json
import os
import gym
import irp.experiments.tile_coding.q as q
import irp.experiments.tile_coding.env as env
from irp.experiments.tile_coding.policy import TiledQTable
import numpy as np
import irp.utils
import irp.envs as envs
import matplotlib.pyplot as plt
import irp.wrappers as wrappers
import gym.wrappers
import irp.q
from sklearn.model_selection import ParameterGrid

# Hyperparameters
parameters = {
    'learning_delay': 0,  # Delay until epsilon starts updating
    'episodes': 5000,       # Total number of episodes
    'alpha': 0.6,           # Learning rate
    'gamma': 0.95,           # Discount factor
    'epsilon': 1.0,         # Amount of randomness in the action selection
    'epsilon_decay': 0.001, # Fixed amount to decrease
    'tilings': 16,          # Number of tilings to use
    'n_thresholds': 2,
    'hash_size': 2**12,
    'min_epsilon': 0.05
}

grid = ParameterGrid({
    'n_thresholds': [5],
    'overlap': [0.5],
    'n_size': [2],
    'tilings': [16]
})

if __name__ == "__main__":
    results = {}

    for param in grid:
        results[str(param)] = {}

        train_name = 'case10_10.png'
        subimage_width, subimage_height = 16, 8
        shape = (512, 512)

        # Specified by the parameter grid
        n_size = param['n_size']
        overlap = param['overlap']
        tilings = param['tilings']
        n_thresholds = param['n_thresholds']

        # Define coordinates of subimages on the border of the label
        coords = [
            (208, 208)
        ]
        coord = coords[0]

        # Get all the training subimages
        subimages, sublabels = np.asarray(
            irp.utils.make_sample_label(train_name, width=subimage_width, height=subimage_height, overlap=overlap, idx=None)
        )[0]

        # Get the subimages in the neighborhood of the image we're analyzing
        n_subimages, n_sublabels = irp.utils.get_neighborhood_images(
            subimages, sublabels, coord, shape, subimage_width, subimage_height, overlap, n_size
        )

        # Create the hashing table for storing observations
        iht = wrappers.utils.IHT(parameters['hash_size'])

        # Create a MultiSample environment to train on multiple subimages
        environments = wrappers.MultiSample([])
        
        for subimage, sublabel in zip(n_subimages, n_sublabels):
            environment = env.Env(subimage, sublabel, n_thresholds)
            environment = wrappers.Tiled(environment, lows=(0, 0, 0), highs=(1, 1, 32), tilings=tilings, iht=iht, rescale=True)
            environment = TimeLimit(environment, environment.n_thresholds * 2)

            environments.add(environment)

        qtable = q.learn(environments, parameters, log=True)

        environment = environments[0]

        s = environment.reset(ti=0)

        for j in range(environment.n_thresholds + 1):
            a = np.argmax(qtable.qs(s))
            s, r, d, info = environment.step(a)

            print(info['d_sim'], environment.d_sim)

        idx = irp.utils.coord_to_id(coord, shape, subimage_width, subimage_height, overlap)

        # Define the test filename and get all the subimages
        test_name = 'case10_11.png'
        subimage, sublabel = np.asarray(
            irp.utils.make_sample_label(test_name, width=subimage_width, height=subimage_height, overlap=overlap, idx=idx)
        )[0]

        environment = env.Env(subimage, sublabel, n_thresholds)
        environment = wrappers.Tiled(environment, lows=(0, 0, 0), highs=(1, 1, 32), tilings=tilings, iht=iht, rescale=True)

        print("Best obtainable dissimilarity:", environment.d_sim)

        s = environment.reset(ti=environment.n_thresholds - 1)

        for j in range(environment.n_thresholds + 2):
            a = np.argmax(qtable.qs(s))
            s, r, d, info = environment.step(a)

            print(info['d_sim'])

        # irp.utils.show(environment.bitmask)

        if np.isclose(info['d_sim'], environment.d_sim):
            print('Solved')