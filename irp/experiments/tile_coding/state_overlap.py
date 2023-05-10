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

grid = ParameterGrid({
    'n_thresholds': [3, 4, 5],
    'overlap': [0, 0.5, 0.75, 0.875],
    'n_size': [1, 2],
})

if __name__ == "__main__":
    for param in grid:
        print(param)

        d_sims = []
        intersections = []

        for coord in [(272, 176), (304, 288), (336, 248), (320, 200)]:
            train_name = 'case10_10.png'
            subimage_width, subimage_height = 16, 8
            overlap = param['overlap']
            shape = (512, 512)
            n_size = param['n_size']

            # Hyperparameters
            parameters = {
                'learning_delay': 250,  # Delay until epsilon starts updating
                'episodes': 5000,       # Total number of episodes
                'alpha': 0.6,           # Learning rate
                'gamma': 0.9,           # Discount factor
                'epsilon': 1.0,         # Amount of randomness in the action selection
                'epsilon_decay': 0.001, # Fixed amount to decrease
                'tilings': 16,          # Number of tilings to use
                'n_thresholds': 4,
                'hash_size': 2**12,
                'min_epsilon': 0.05
            }

            tilings = parameters['tilings']
            n_thresholds = param['n_thresholds']
            iht = wrappers.utils.IHT(parameters['hash_size'])
            
            # coord = (272, 176)
            # coord = (304, 288)
            # coord = (336, 248)
            # coord = (320, 200)
            idx = irp.utils.coord_to_id(coord, shape, subimage_width, subimage_height, overlap)

            # Get all the training subimages
            subimages, sublabels = np.asarray(
                irp.utils.make_sample_label(train_name, width=subimage_width, height=subimage_height, overlap=overlap, idx=None)
            )[0]

            # Get the subimages in the neighborhood of the image we're analyzing
            n_subimages, n_sublabels = irp.utils.get_neighborhood_images(
                subimages, sublabels, coord, shape, subimage_width, subimage_height, overlap, n_size
            )

            states = set()
            training_overlap = []

            # Collect train states
            for subimage, sublabel in zip(n_subimages, n_sublabels):
                environment = env.Env(subimage, sublabel, n_thresholds)
                environment = wrappers.Tiled(environment, lows=(0, 0, 0), highs=(1, 1, 32), tilings=tilings, iht=iht, rescale=True)

                # states__ = set()

                for i in range(environment.n_thresholds):
                    state = environment.reset(ti=i)

                    states.add(tuple(state.tolist()))
                    # states__.add(tuple(state.tolist()))

                # training_overlap.append(set(np.asarray(list(states__)).flatten().tolist()))

            flat_states = set(np.asarray(list(states)).flatten().tolist())

            # print("Training overlap", len(set.intersection(*training_overlap)) / len(flat_states))

            # Define the test filename and get all the subimages
            test_name = 'case10_11.png'
            subimage, sublabel = np.asarray(
                irp.utils.make_sample_label(test_name, width=subimage_width, height=subimage_height, overlap=overlap, idx=idx)
            )[0]

            environment = env.Env(subimage, sublabel, n_thresholds)
            environment = wrappers.Tiled(environment, lows=(0, 0, 0), highs=(1, 1, 32), tilings=tilings, iht=iht, rescale=True)

            best_d_sim, ti = irp.utils.get_best_dissimilarity(subimage, sublabel, environment.intensity_spectrum, return_ti=True)

            # print("Best obtainable d-sim:", best_d_sim)

            # irp.utils.show(np.hstack([sublabel, envs.utils.apply_threshold(subimage, ti)]))

            states_ = set()

            for i in range(environment.n_thresholds):
                state = environment.reset(ti=i)

                states_.add(tuple(state.tolist()))


            # print(len(flat_states))

            flat_states_ = set(np.asarray(list(states_)).flatten().tolist())

            # print(len(flat_states_))

            # print(len(flat_states & flat_states_) / len(flat_states_))

            d_sims.append(best_d_sim)
            intersections.append(len(flat_states & flat_states_) / len(flat_states_))

        print('Average and std d_sim', round(np.mean(d_sims), 2), round(np.std(d_sims), 4))
        print('Average and std overlap', round(np.mean(intersections), 2), round(np.std(intersections), 4))
        print()

            # print(len(states), len(states_))
            # [print(state) for state in states]