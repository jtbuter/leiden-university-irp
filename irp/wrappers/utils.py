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

basehash = hash

class IHT:
    "Structure to handle collisions"
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
            if self.overfullCount == 0: raise ValueError('IHT full, starting to allow collisions')
            
            self.overfullCount += 1

            return basehash(obj) % self.size
        else:
            d[obj] = count
            
            return count

def hashcoords(
    coordinates: Tuple[float, ...],
    m: Union[IHT, int, None],
    readonly: bool = False
) -> Union[Tuple[float, ...], int]:
    if type(m)==IHT: return m.getindex(tuple(coordinates), readonly)
    if type(m)==int: return basehash(tuple(coordinates)) % m
    if m==None: return coordinates

def tiles(
    ihtORsize: Union[IHT, int, None],
    numtilings: int,
    floats: List[float],
    ints: List[int] = [],
    readonly: bool = False
) -> Union[List[int], List[Tuple[float, ...]]]:
    """returns num-tilings tile indices corresponding to the floats and ints"""
    qfloats = [math.floor(f * numtilings) for f in floats]
    Tiles = []

    for tiling in range(numtilings):
        tilingX2 = tiling * 2
        coords = [tiling]
        b = tiling

        for q in qfloats:
            coords.append((q + b) // numtilings )
            b += tilingX2

        coords.extend(ints)
        Tiles.append(hashcoords(coords, ihtORsize, readonly))

    return Tiles

def min_max_scaling(x, mi, ma):
    # Assumes we're working with Sutton tile-coding (i.e. division of 10.0)
    return np.clip((x - mi) * (10.0 / (ma - mi)), 0, 10.0)

#!/usr/bin/env python
import numpy as np

class TileCoder:
    def __init__(self, tiles_per_dim, value_limits, tilings, offset=lambda n: 2 * np.arange(n) + 1):
        tiling_dims = np.array(np.ceil(tiles_per_dim), dtype=np.int) + 1
        self._offsets = offset(len(tiles_per_dim)) * np.repeat([np.arange(tilings)], len(tiles_per_dim), 0).T / float(tilings) % 1

        self._limits = np.array(value_limits)
        self._norm_dims = np.array(tiles_per_dim) / (self._limits[:, 1] - self._limits[:, 0])
        self._tile_base_ind = np.prod(tiling_dims) * np.arange(tilings)
        self._hash_vec = np.array([np.prod(tiling_dims[0:i]) for i in range(len(tiles_per_dim))])
        self._tiling_dims = np.array(np.ceil(tiles_per_dim), dtype=np.int)
        self._n_tiles = tilings * np.prod(tiling_dims)

    def __getitem__(self, x):
        off_coords = ((x - self._limits[:, 0]) * self._norm_dims + self._offsets).astype(int)
        return self._tile_base_ind + np.dot(off_coords, self._hash_vec)

    def tile(self, x):
        # x = np.clip(x, 0, 1 - (1 / self._tiling_dims) / len(self._offsets))

        off_coords = ((x - self._limits[:, 0]) * self._norm_dims + self._offsets).astype(int)

        # off_coords = np.clip(off_coords, 0, self._tiling_dims_m1)

        # print(off_coords)

        # return off_coords

        return self._tile_base_ind + np.dot(off_coords, self._hash_vec)

    @property
    def n_tiles(self):
        return self._n_tiles
