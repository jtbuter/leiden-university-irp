from typing import List, Tuple, Union

import numpy as np
import gym.spaces

def discrete(
    sample: Union[Tuple, List, np.ndarray],
    grid: List[np.ndarray]
) -> Tuple[int, ...]:
    return tuple(int(np.searchsorted(g, s)) for s, g in zip(sample, grid))

def get_dims(*args: List[gym.spaces.Space]):
    dims = tuple()

    for space in args:
        if isinstance(space, gym.spaces.MultiDiscrete):
            dims += tuple(space.nvec)
        else:
            dims += (space.n,)

    return dims
