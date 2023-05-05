import numpy as np
from typing import Tuple
from irp.wrappers import Discretize
import timeit

def discrete__digitize(sample, grid) -> Tuple[int, ...]:
    return tuple(int(np.digitize(s, g)) for s, g in zip(sample, grid))

def discrete__searchsorted(sample, grid) -> Tuple[int, ...]:
    return tuple(int(np.searchsorted(g, s)) for s, g in zip(sample, grid))

grid = Discretize.make_state_bins((4, 4, 3), (0, 0, 1), (1, 1, 3))
sample = (0.34, 0.1, 1)

print(timeit.timeit(lambda: discrete__digitize(sample, grid), number=100000))
print(timeit.timeit(lambda: discrete__searchsorted(sample, grid), number=100000))

# print(discrete__digitize(sample, grid))
# print(discrete__searchsorted(sample, grid))