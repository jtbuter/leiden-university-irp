import numpy as np
import random

from sklearn.model_selection import ParameterGrid

image_parameters = {
    'dimensions': lambda: random.choice([(16, 8), (32, 16)])
}
tiling_param_grid = {
    'tiles_per_dim': lambda: random.choice([(2, 2, 2), (4, 4, 4)]),
    'tilings': lambda: random.choice([16, 32, 64]),
    'limits': lambda: ((0, 1), (0, 1), (0, 32))
}
agent_param_grid = {
    'alpha': lambda: random.choice(np.arange(0.1, 0.6, 0.05)),
    'max_t': lambda: random.choice([2000, 5000, 10000]),
    'max_e': lambda: np.inf,
    'eps_max': lambda: random.choice(np.arange(0.1, 0.8, 0.05)),
    'eps_frac': lambda: 1.0,
    'gamma': lambda: random.choice([0.6, 0.7, 0.8, 0.9, 0.99]),
}
environment_parameters = {
    'n_thresholds': lambda: 5,
    'openings': lambda: random.choice([[0], [0, 2, 5]]),
    'ranged': lambda: random.choice([False, True]),
    # 'sahba': lambda: random.choice([False, True])
}

param_grid = {**image_parameters, **tiling_param_grid, **agent_param_grid, **environment_parameters}

for i in range(200):
    for key, value_fn in param_grid.items():
        print(key, value_fn())

    print()
