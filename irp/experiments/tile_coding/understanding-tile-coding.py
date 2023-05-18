# Import common libraries
import sys
import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import irp.wrappers as wrappers

def create_tiling_grid(low, high, tiles_per_dim, offsets=(0.0, 0.0)):
    grid = [
        np.linspace(low[dim], high[dim], tiles_per_dim + 1)[1:-1] + offsets[dim] for dim in range(len(offsets))
    ]

    return grid

def create_tilings(low, high, tiling_specs):
    return [
        create_tiling_grid(low, high, tiles_per_dim, offsets) for tiles_per_dim, offsets in tiling_specs
    ]

def discretize(sample, grid):
    return tuple(int(np.digitize(s, g)) for s, g in zip(sample, grid))  # apply along each dimension

def tile_encode(sample, tilings, flatten=False):
    encoded_sample = [discretize(sample, grid) for grid in tilings]

    return np.concatenate(encoded_sample) if flatten else encoded_sample

def tile_to_hash(coordinates, tiles_per_dim):
    indices = []
    dims = len(coordinates[0])

    for ntl, coordinate in enumerate(coordinates):
        total_tiles = tiles_per_dim ** dims

        tiling_index = 0

        for dim, coord in enumerate(coordinate):
            tiling_index += coord * tiles_per_dim ** dim

        indices.append(tiling_index + total_tiles * ntl)

    return indices

low = [0.0]
high = [1.0]
dims = len(low)

tiles_per_dim = 2

# Tiling specs: [(<bins>, <offsets>), ...]
tiling_specs = [
    (tiles_per_dim, [0.0]),
    # (tiles_per_dim, [-0.24]),
    # ((10, 10), (0.066, 0.33))
]

n_tilings = len(tiling_specs)
tilings = create_tilings(low, high, tiling_specs)

print('w/n =', 1 / tiles_per_dim / n_tilings)

print(tile_encode((0.0,), tilings))
print(tile_encode((0.25,), tilings))
print(tile_encode((0.5,), tilings))
print(tile_encode((0.75,), tilings))
print(tile_encode((1.0,), tilings))
# print(tile_encode((0.5, 0.0), tilings))
# print(tile_encode((0.0, 0.5), tilings))
# print(tile_encode((0.5, 0.5), tilings))
# print(tile_encode((0.5, 0.75), tilings))

# print(tile_to_hash(tile_encode((0.0, 0.0), tilings), tiles_per_dim))
# print(tile_to_hash(tile_encode((0.5, 0.0), tilings), tiles_per_dim))

T = wrappers.utils.TileCoder((tiles_per_dim,) * dims, list(zip(low, high)), n_tilings)

print()

print(0.0, T.tile((0.0)))
print(0.25, T.tile((0.25)))
print(0.5, T.tile((0.5)))
print(0.75, T.tile((0.75)))
print(1.0, T.tile((1.0)))
# print(wrappers.utils.TileCoder((tiles_per_dim,) * dims, [[0.0, 1.0], [0.0, 1.0]], n_tilings).tile((0.5, 0.0)))
# print(wrappers.utils.TileCoder((tiles_per_dim,) * dims, [[0.0, 1.0], [0.0, 1.0]], n_tilings).tile((0.0, 0.5)))
# print(wrappers.utils.TileCoder((tiles_per_dim,) * dims, [[0.0, 1.0], [0.0, 1.0]], n_tilings).tile((0.5, 0.5)))