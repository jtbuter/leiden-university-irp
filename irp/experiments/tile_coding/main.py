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

if __name__ == "__main__":
    grid = ParameterGrid({
        'n_thresholds': [3],
        'overlap': [0.75],
        'n_size': [1],
        'tilings': [16]
    })

    results = {}

    for param in grid:
        print(param)

        results[str(param)] = {}

        train_name = 'case10_10.png'
        subimage_width, subimage_height = 16, 8
        overlap = param['overlap']
        shape = (512, 512)
        n_size = param['n_size']

        # Hyperparameters
        parameters = {
            'learning_delay': 500,  # Delay until epsilon starts updating
            'episodes': 2000,       # Total number of episodes
            'alpha': 0.6,           # Learning rate
            'gamma': 0.9,           # Discount factor
            'epsilon': 1.0,         # Amount of randomness in the action selection
            'epsilon_decay': 0.001, # Fixed amount to decrease
            'tilings': 16,          # Number of tilings to use
            'n_thresholds': 5,
            'hash_size': 2**12,
            'min_epsilon': 0.05
        }

        tilings = param['tilings']
        n_thresholds = param['n_thresholds']

        coords = [(192, 176), (256, 232), (304, 288), (336, 248), (272, 176)]
        coords = [(256, 224)]

        for coord in coords:
            solved = []
        
            for xy in range(3):
                solved.append(0)
            
                iht = wrappers.utils.IHT(parameters['hash_size'])
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
                    environment = wrappers.Tiled(environment, lows=(0, 0, 0), highs=(1, 1, 32), tilings=tilings, iht=iht, rescale=True)

                    environments.add(environment)

                qtable = q.learn(environments, parameters, log=True)

                # Define the test filename and get all the subimages
                test_name = 'case10_11.png'
                subimage, sublabel = np.asarray(
                    irp.utils.make_sample_label(test_name, width=subimage_width, height=subimage_height, overlap=overlap, idx=idx)
                )[0]

                environment = env.Env(subimage, sublabel, n_thresholds)
                environment = gym.wrappers.TimeLimit(environment, 30) # TODO: Kunnen we deze weghalen uiteindelijk
                environment = wrappers.Tiled(environment, lows=(0, 0, 0), highs=(1, 1, 32), tilings=tilings, iht=iht, rescale=True)

                # print("Best obtainable dissimilarity:", environment.d_sim)

                s = environment.reset(ti=environment.n_thresholds - 1)

                for j in range(environment.n_thresholds + 2):
                    a = np.argmax(qtable.qs(s))
                    s, r, d, info = environment.step(a)

                    # print(info['d_sim'])

                print(environment.ti)
                irp.utils.show(environment.bitmask)

                if np.isclose(info['d_sim'], environment.d_sim):
                    solved[-1] = 1

            results[str(param)][str(coord)] = solved

            print(coord, solved)

            # with open(os.path.join(irp.ROOT_DIR, 'results/tile_coding', 'data.json'), 'w') as f:
            #     json.dump(results, f, ensure_ascii=False, indent=4)
