from typing import List, Optional, Tuple, Union
import numpy as np

import gym

import irp.experiments.tile_coding.env as env
import irp.wrappers as wrappers

class TiledQTable():
    def __init__(self, environment: gym.Env, tilings: int):
        self.tilings = tilings
        self.qtable = self._build_qtable(environment)

    def _build_qtable(self, environment: gym.Env) -> np.ndarray:
        dims = wrappers.utils.get_dims(environment.observation_space, environment.action_space)

        return np.zeros(dims)

    def value(self, state: List[int], action: Optional[int] = None) -> float:
        value = 0.0

        for tile in state:
            if action:
                value += self.qtable[tile, action]
            else:
                value += np.max(self.qtable[tile])

        return value

    def update(self, state: List[int], action: int, target: float, alpha: float):
        estimate = 0.0

        for tile in state:
            estimate += self.qtable[tile, action]

        error = target - estimate

        for tile in state:
            self.qtable[tile, action] += alpha * error

        # for tile in state: # TODO Deze manier weer gebruiken
        #     self.qtable[tile, action] += alpha * (target - self.qtable[tile, action])


if __name__ == "__main__":
    def target_fn(x, y):
        return np.sin(x) + np.cos(y) + 0.1 * np.random.randn()

    class PiEnv():
        def __init__(self, iht):
            self.observation_space = gym.spaces.Discrete(n=iht.size)
            self.action_space = gym.spaces.Discrete(n=1)

    # Hyperparameters
    parameters = {
        'alpha': 0.1,           # Learning rate
        'tilings': 8,           # Number of tilings to use
        'hash_size': 2048
    }

    iht = wrappers.utils.IHT(parameters['hash_size'])
    tilings = parameters['tilings']


    alpha = parameters['alpha'] / tilings
    action = 0

    environment = PiEnv(iht)
    environment = wrappers.Tiled(environment, lows=(0, 0), highs=(2 * np.pi, 2 * np.pi), tilings=tilings, iht=iht, rescale=False)

    error = []
    qtable = TiledQTable(environment, tilings)

    for i in range(1000):
        # get noisy sample from target function at random location
        x, y = 2.0 * np.pi * np.random.rand(2)
        target = target_fn(x, y)
        state = environment.encode(iht, tilings, (x, y), rescale=False, input_limits=environment._input_limits)
        
        qtable.update(state, 0, target, alpha)

    x, y = 2.5, 3.1
    state = state = environment.encode(iht, tilings, (x, y), rescale=False, input_limits=environment._input_limits)

    print((np.sin(x) + np.cos(y)), qtable.value(state))

    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D

    # # resolution
    # res = 200

    # # (x, y) space to evaluate
    # x = np.arange(0.0, 2.0 * np.pi, 2.0 * np.pi / res)
    # y = np.arange(0.0, 2.0 * np.pi, 2.0 * np.pi / res)

    # # map the function across the above space
    # z = np.zeros([len(x), len(y)])
    # for i in range(len(x)):
    #     for j in range(len(y)):
    #         state = environment.encode(iht, tilings, (x[i], y[j]))
    #         z[i, j] = qtable.value(state, 0)

    # # plot function
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # X, Y = np.meshgrid(x, y)
    # surf = ax.plot_surface(X, Y, z, cmap=plt.get_cmap('hot'))
    # plt.show()


