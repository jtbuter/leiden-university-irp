import math

from typing import List, Tuple, Union

import numpy as np
import gym.spaces

def discrete(
    sample: Union[Tuple, List, np.ndarray],
    grid: List[np.ndarray]
) -> Tuple[int, ...]:
    return tuple(int(np.searchsorted(g, s)) for s, g in zip(sample, grid))

def get_dims(*args: List[gym.spaces.Space]) -> Tuple[int, ...]:
    dims = tuple()

    for space in args:
        if isinstance(space, gym.spaces.MultiDiscrete):
            dims += tuple(space.nvec)
        else:
            dims += (space.n,)

    return dims

class IHT:
    def __init__(self, size: int):
        self.size = size
        self.overfull_count = 0
        self.dictionary = {}

    def count(self) -> int:
        return len(self.dictionary)

    def fullp(self) -> bool:
        return len(self.dictionary) >= self.size

    def getindex(
        self,
        obj: Tuple[float, ...],
        readonly: bool = False
    ) -> int:
        d = self.dictionary

        if obj in d: return d[obj]
        elif readonly: return None
        
        size = self.size
        count = self.count()

        if count >= size:
            if self.overfull_count == 0: print('IHT full, starting to allow collisions')
        
            self.overfull_count += 1
        
            return hash(obj) % self.size
        else:
            d[obj] = count

            return count

def hashcoords(
    coordinates: Tuple[float, ...],
    m: Union[IHT, int, None],
    readonly: bool = False
) -> Union[Tuple[float, ...], int]:
    if m is None: return coordinates
    if isinstance(m, IHT): return m.getindex(tuple(coordinates), readonly)
    if isinstance(m, int): return hash(tuple(coordinates)) % m

def tiles(
    ihtORsize: Union[IHT, int, None],
    tilings: int,
    positions: List[float],
    actions: List[int] = [],
    readonly: bool = False
) -> Union[List[int], List[Tuple[float, ...]]]:
    qpositions = [math.floor(f * tilings) for f in positions]
    tiles_ = []

    for tiling in range(tilings):
        tilingX2 = tiling * 2
        coords = [tiling]
        b = tiling
        
        for q in qpositions:
            coords.append((q + b) // tilings )

            b += tilingX2

        coords.extend(actions)

        tiles_.append(hashcoords(coords, ihtORsize, readonly))

    return tiles_

def min_max_scaling(x, mi, ma):
    # Assumes we're working with Sutton tile-coding (i.e. division of 10.0)
    return np.clip((x - mi) * (10.0 / (ma - mi)), 0, 10.0)

