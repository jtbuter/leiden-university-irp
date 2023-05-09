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
    # Hyperparameters
    parameters = {
        'learning_delay': 0,    # Delay until epsilon starts updating
        'episodes': 4000,        # Total number of episodes
        'alpha': 0.5,           # Learning rate
        'gamma': 0.9,           # Discount factor
        'epsilon': 1.0,         # Amount of randomness in the action selection
        'epsilon_decay': 0.01, # Fixed amount to decrease
        'tilings': 16,          # Number of tilings to use
        'n_thresholds': 6,
        'hash_size': 2048,
        'min_epsilon': 0.05
    }

    # Deze berekening is raar
    # parameters['hash_size'] = parameters['n_thresholds'] * (parameters['tilings'] + 3)
    
    tilings = parameters['tilings']
    n_thresholds = parameters['n_thresholds']
    iht = wrappers.utils.IHT(parameters['hash_size'])

    multi_environment = wrappers.MultiSample([])

    for idx in [720, 721]:
        sample, label = irp.utils.make_sample_label('case10_10.png', idx=idx)[0]
        
        environment = env.Env(sample, label, n_thresholds)
        environment = gym.wrappers.TimeLimit(environment, 30)
        environment = wrappers.Tiled(environment, lows=(0, 0, 1), highs=(1, 1, 4), tilings=tilings, iht=iht, rescale=True)

        best_d_sim = irp.utils.get_best_dissimilarity(sample, label, n_thresholds)

        print(f'Best d_sim for index {idx}: {best_d_sim}')

        multi_environment.add(environment)

    qtable = q.learn(multi_environment, parameters, log=True)

    success = np.zeros((10,))

    for i in range(success.size):
        qtable = q.learn(multi_environment, parameters, log=False)

        # Assert that we succeeded in learning
        success[i] = True

        s = multi_environment.reset()

        for j in range(10):
            a = np.argmax(qtable.qs(s))
            s, r, d, info = multi_environment.step(a)

        print(info['d_sim'])

        success[i] = np.isclose(info['d_sim'], multi_environment.env.d_sim).astype(int)

    print(f'Success rate: {((success.sum() / success.size) * 100):.2f} %')