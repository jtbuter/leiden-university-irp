import json
from irp.wrappers.utils import tiles, IHT
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, TypeVar

def getAxisCell(bound: str, x: float, tiles_per_dim: int):
    # for a 2-d space, this would get the "row" then "col" for a given coordinate
    # for example: pos = (0.1, 0.3) with 5 tiles per dim would give row=1 and col=2
    i = int(np.floor(x * tiles_per_dim))

    if bound == 'wrap':
        return i % tiles_per_dim
    elif bound == 'clip':
        return clip(i, 0, tiles_per_dim - 1)

    raise Exception('Unknown bound type')

def getTileIndex(dimensions: int, tiles_per_dim: int, coords: List[float]) -> int:
    # the index of the tile that coords is in
    ind = 0
    tile_length = 1 / tiles_per_dim

    for dim in range(dimensions):
        # In this particular dimension, find out how far from 0 we are and how close to 1
        d1_tile_index = clip(coords[dim] // tile_length, 0, tiles_per_dim - 1)

        # Then offset that index by the number of tiles in all dimensions before this one.
        ind += d1_tile_index * (tiles_per_dim ** dim)

    return ind

def getTilingIndex(bound: str, dims: int, tiles_per_dim: int, pos):
    ind = 0
    total_tiles = tiles_per_dim ** dims
    
    for d in range(dims):
        # which cell am I in on this axis?
        axis = getAxisCell(bound, pos[d], tiles_per_dim)
        already_seen = tiles_per_dim ** d
        ind += axis * already_seen

    # ensure we don't overflow into another tiling
    return clip(ind, 0, total_tiles - 1)

def getTCIndices(dims: int, tiles: int, tilings: int, bound: str, offsets, pos, action: Optional[int] = None):
    total_tiles = tiles**dims

    index = np.empty((tilings), dtype='int64')
    for ntl in range(tilings):
        ind = getTilingIndex(bound, dims, tiles, pos + offsets[ntl])
        index[ntl] = ind + total_tiles * ntl

    if action is not None:
        index += action * total_tiles * tilings

    return index

def minMaxScaling(x, mi, ma):
    return (x - mi) / (ma - mi)

def clip(x, mi, ma):
    return max(min(x, ma), mi)

def create_tiling_grid(low, high, bins=(2, 2), offsets=(0.0, 0.0)):
    grid = [np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] + offsets[dim] for dim in range(len(bins))]

    return grid

def discretize(sample, grid):
    return tuple(int(np.digitize(s, g)) for s, g in zip(sample, grid))  # apply along each dimension


def tile_encode(sample, tiles_per_dim, tilings, flatten=False):
    index = 0

    for ntl, grid in enumerate(tilings):
        encoded_sample = discretize(sample, grid)
        tiling_index = 0

        for dim, coord in enumerate(encoded_sample):
            tiling_index += coord * tiles_per_dim ** dim

        index += tiling_index + ntl * (len(tilings) ** len(sample))

    return index

tile_index_vals = []
tilings_index_vals = []
tile_encode_index_vals = []
tiles3_index_vals = []

dims = 2
tiles_per_dim = 2
grid = [create_tiling_grid((0.,) * dims, (1.,) * dims, (tiles_per_dim,) * dims, (0.,) * dims)]
iht = IHT(10000)

for x in range(1, 10):
    x /= 10

    for y in range(1, 10):
        y /= 10

        pos = [x, y]

        tile_index = int(getTileIndex(dims, tiles_per_dim, pos))
        tilings_index = getTilingIndex('clip', dims, tiles_per_dim, pos)
        # tile_encode_index_x, tile_encode_y, tile_encode_z = tile_encode(pos, grid, flatten=True)
        # tile_encode_index = tile_encode_index_x + tile_encode_y * 2 + tile_encode_z * 4
        tile_encode_index = tile_encode(pos, tiles_per_dim, grid, flatten=True)
        # tile_encode_index = tile_encode_index_x + tile_encode_y * 2

        tile_index_vals.append(tile_index)
        tilings_index_vals.append(tilings_index)
        tile_encode_index_vals.append(tile_encode_index)

results = [
    np.unique(tile_index_vals, return_counts=True),
    np.unique(tilings_index_vals, return_counts=True),
    np.unique(tile_encode_index_vals, return_counts=True)
]

# pos = [1, 1]

# print(tiles(ihtORsize=None, tilings=1, positions=[pos[0] * tiles_per_dim, pos[1] * tiles_per_dim]))
# print(tile_encode(pos, grid, flatten=True))

print('\n'.join(map(str, results)))