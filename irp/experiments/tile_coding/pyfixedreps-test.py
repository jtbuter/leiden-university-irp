import numpy as np
from PyFixedReps import TileCoder

tc = TileCoder({
    # [required]
    'tiles': 2, # how many tiles in each tiling
    'tilings': 4,
    'dims': 2, # shape of the state-vector
    'input_ranges': [(0, 2 * np.pi), (0, 2 * np.pi)], # a vector of same length as 'dims' containing (min, max) tuples to rescale inputs
    # 'scale_output': True, # scales the output by number of active features
})
state = [0.0, 0.0]
indices = tc.get_indices(state)
