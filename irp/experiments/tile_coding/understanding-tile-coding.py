import json
from irp.wrappers.utils import tiles, IHT
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, TypeVar

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

def getTilingsIndices(dimensions: int, tiles: int, tilings: int, coords: List[float]) -> List[int]:
    tiles_per_tiling = tiles**dimensions
    tile_length = 1 / tiles

    indices = np.empty(tilings)

    for tiling_nb in range(tilings):
        offset = tile_length * tiling_nb / tilings
        offset = 0

        # because this wraps around when the inputs are
        # bigger than 1, this is a safe operation
        # ind = getTileIndex(dimensions, tiles, coords + offset)
        ind = getTileIndex(dimensions, tiles, coords)

        # store the index, but first offset it by the number
        # of tiles in all tilings before us
        indices[tiling_nb] = ind + tiles_per_tiling * tiling_nb

    return indices

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

def getTCIndices(dims: int, tiles: int, tilings: int, bound: str, offsets: np.ndarray, pos: np.ndarray, action: Optional[int] = None):
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
    tilings = [discretize(sample, grid) for grid in tilings]
    print(tilings)
    indices = coordinates_to_index(tilings, tiles_per_dim)

    return indices

def coordinates_to_index(coordinates, tiles_per_dim):
    indices = []
    dims = len(coordinates[0])

    for ntl, coordinate in enumerate(coordinates):
        total_tiles = tiles_per_dim ** dims

        tiling_index = 0

        for dim, coord in enumerate(coordinate):
            tiling_index += coord * tiles_per_dim ** dim

        indices.append(tiling_index + total_tiles * ntl)

    return indices

def build_offset(n: int, dims, tiles, tilings, offset: str = 'even'):
    if offset == 'cascade':
        tile_length = 1.0 / tiles
        return np.ones(dims) * n * (tile_length / tilings)

    if offset == 'even':
        tile_length = 1.0 / tiles
        i = n - (tilings / 2)
        return np.ones(dims) * i * (tile_length / tilings)

tile_index_vals = []
tilings_index_vals = []
tile_encode_index_vals = []
tiles3_index_vals = []

dims = 3
tiles_per_dim = 2
tilings = 16
offsets = [(i/tilings,) * dims for i in range(tilings)]
grid = [create_tiling_grid((0.,) * dims, (1.,) * dims, (tiles_per_dim,) * dims, offsets=offsets[i]) for i in range(tilings)]
iht = IHT(100000)

pos = (0.6,)

mi, ma = 0.0, 1.0
w = ((ma - mi) / tiles_per_dim)
n = tilings
wn = w / n 
stepsize = int(round(1 / wn, 0))

# Meer algmene formule Sutton
print(f'w = {w}, n = {n} -> w/n =', wn)

multiplier = round(1 / wn / n, 0)

tile_extremes: Dict[int, Dict[str, List[Tuple]]] = {}

for x in np.linspace(mi, ma, stepsize, endpoint=False, dtype=np.float32):
    for y in np.linspace(mi, ma, stepsize, endpoint=False, dtype=np.float32):
        for z in np.linspace(mi, ma, stepsize, endpoint=False, dtype=np.float32):
            x = round(x, 5)
            y = round(y, 5)
            z = round(z, 5)

            scaled_x = x * (tiles_per_dim / (ma - mi))
            scaled_y = y * (tiles_per_dim / (ma - mi))
            scaled_z = z * (tiles_per_dim / (ma - mi))

            result = tiles(None, numtilings=tilings, floats=[scaled_x, scaled_y, scaled_z])

            for tile_idx, tile_coord_1, tile_coord_2, tile_coord_3 in result:
                tile_idx, tile_coord_1, tile_coord_2, tile_coord_3 = int(tile_idx), int(tile_coord_1), int(tile_coord_2), int(tile_coord_3)

                if tile_idx not in tile_extremes:
                    tile_extremes[tile_idx] = {}

                coord_name = str((tile_coord_1, tile_coord_2, tile_coord_3))

                if coord_name not in tile_extremes[tile_idx]:
                    tile_extremes[tile_idx][coord_name] = []

                tile_extremes[tile_idx][coord_name].append((float(x), float(y), float(z)))


        # print(x, y, result)

base_x, base_y, base_z = tile_extremes[0]['(0, 0, 0)'][-1]

# print(tile_extremes)
for tile_idx in tile_extremes:
    print(tile_idx)

    max_x, max_y, max_z = tile_extremes[tile_idx][next(iter(tile_extremes[tile_idx]))][-1]

    print('\t', (base_x - max_x) / (wn), (base_y - max_y) / (wn), (base_z - max_z) / (wn))

    # for coord_name in tile_extremes[tile_idx]:
        # print(f'\t{coord_name}')
        # print(f'\t\t{tile_extremes[tile_idx][coord_name][-1]}')
        # print()


offsets = np.asarray([[0, 0], [0.5, 0.5]])
offsets = [build_offset(n, dims, tiles_per_dim, tilings) for n in range(tilings)]

print(offsets)

# tc_index = getTCIndices(dims, tiles_per_dim, tilings, 'clip', offsets, np.asarray(pos))


# pos = (0.06,)

# tiling_index = getTilingsIndices(dims, tiles_per_dim, tilings, pos)
# tc_index = getTCIndices(dims, tiles_per_dim, tilings, 'clip', np.zeros((tilings, dims)), np.asarray(pos))
# tile_encode_index = tile_encode(pos, tiles_per_dim, grid)
# tiles3_index_ = np.asarray(tiles(None, tilings=tilings, positions=[pos[0] * 10 * tiles_per_dim]))[:, 1:].flatten()
# tiles3_index_ = tiles(None, tilings=tilings, positions=[pos[0] * 10 * tiles_per_dim])
# tiles3_index = coordinates_to_index(np.asarray(tiles(None, tilings=tilings, positions=[pos[0] * 10]))[:, 1:], tiles_per_dim)

# # for state in np.arange(0, 2.5, 0.15):
# #     indices = np.asarray(tiles(None, tilings=2, positions=[state]))[:, 1:].flatten()

# #     print('{0:.2f}'.format(state), ' -> ', indices)

# print(tiling_index)
# print(tc_index)
# print(tile_encode_index)
# print(tiles3_index_)
# print(tiles3_index)

# for x in range(1, 10):
#     x /= 10

#     for y in range(1, 10):
#         y /= 10

#         pos = [x, y]

#         tile_index = int(getTileIndex(dims, tiles_per_dim, pos))
#         tilings_index = getTilingIndex('clip', dims, tiles_per_dim, pos)
#         tile_encode_index = tile_encode(pos, tiles_per_dim, grid, flatten=True)

#         tile_index_vals.append(tile_index)
#         tilings_index_vals.append(tilings_index)
#         tile_encode_index_vals.append(tile_encode_index)

# results = [
#     np.unique(tile_index_vals, return_counts=True),
#     np.unique(tilings_index_vals, return_counts=True),
#     np.unique(tile_encode_index_vals, return_counts=True)
# ]

# # pos = [1, 1]

# # print(tiles(ihtORsize=None, tilings=1, positions=[pos[0] * tiles_per_dim, pos[1] * tiles_per_dim]))
# # print(tile_encode(pos, grid, flatten=True))

# print('\n'.join(map(str, results)))