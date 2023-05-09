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

if __name__ == "__main__":
    train_name = 'case10_10.png'
    subimage_width, subimage_height = 16, 8
    overlap = 0.75
    shape = (512, 512)
    n_size = 3

    # Hyperparameters
    parameters = {
        'learning_delay': 1000, # Delay until epsilon starts updating
        'episodes': 2000,      # Total number of episodes
        'alpha': 0.5,           # Learning rate
        'gamma': 0.9,           # Discount factor
        'epsilon': 1.0,         # Amount of randomness in the action selection
        'epsilon_decay': 0.001, # Fixed amount to decrease
        'tilings': 48,          # Number of tilings to use
        'n_thresholds': 6,
        'hash_size': 2**12,
        'min_epsilon': 0.05
    }

    tilings = parameters['tilings']
    n_thresholds = parameters['n_thresholds']
    iht = wrappers.utils.IHT(parameters['hash_size'])
    
    coord = (272, 176)
    idx = irp.utils.coord_to_id(coord, shape, subimage_width, subimage_height, overlap)

    # Get all the training subimages
    subimages, sublabels = np.asarray(
        irp.utils.make_sample_label(train_name, width=subimage_width, height=subimage_height, overlap=overlap, idx=None)
    )[0]

    # Get the subimages in the neighborhood of the image we're analyzing
    n_subimages, n_sublabels = irp.utils.get_neighborhood_images(
        subimages, sublabels, coord, shape, subimage_width, subimage_height, overlap, n_size
    )

    # # Create a MultiSample environment to train on multiple subimages
    environments = wrappers.MultiSample([])

    for subimage, sublabel in zip(n_subimages, n_sublabels):
        environment = env.Env(subimage, sublabel, n_thresholds)
        environment = gym.wrappers.TimeLimit(environment, 30) # TODO: Kunnen we deze weghalen uiteindelijk
        environment = wrappers.Tiled(environment, lows=(0, 0, 1), highs=(1, 1, 4), tilings=tilings, iht=iht, rescale=True)

        environments.add(environment)

    qtable = q.learn(environments, parameters, log=True)

    # Define the test filename and get all the subimages
    test_name = 'case10_11.png'
    subimage, sublabel = np.asarray(
        irp.utils.make_sample_label(test_name, width=subimage_width, height=subimage_height, overlap=overlap, idx=idx)
    )[0]

    environment = env.Env(subimage, sublabel, n_thresholds)
    environment = gym.wrappers.TimeLimit(environment, 30) # TODO: Kunnen we deze weghalen uiteindelijk
    environment = wrappers.Tiled(environment, lows=(0, 0, 1), highs=(1, 1, 4), tilings=tilings, iht=iht, rescale=True)

    print("Best obtainable dissimilarity:", environment.d_sim)

    s = environment.reset(ti=n_thresholds - 1)

    print(qtable.qs(s))

    for j in range(10):
        a = np.argmax(qtable.qs(s))
        s, r, d, info = environment.step(a)

        print(info['d_sim'])
