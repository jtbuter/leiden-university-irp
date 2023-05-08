import numpy as np
from typing import List

# construct tiling offsets
# defaults to evenly spaced tilings
def build_offset(dims, tiles, tilings, n: int):
    tile_length = 1.0 / tiles

    return np.ones(dims) * n * (tile_length / tilings)

def get_indices(dims, tiles, tilings, offsets, pos: List, input_ranges):
    pos_ = np.asarray(pos, dtype=np.float32)

    if input_ranges is not None:
        pos_ = minMaxScaling(pos_, input_ranges[:, 0], input_ranges[:, 1])

    return getTCIndices(dims, tiles, tilings, offsets, pos_)

def getAxisCell(x: float, tiles: int):
    # for a 2-d space, this would get the "row" then "col" for a given coordinate
    # for example: pos = (0.1, 0.3) with 5 tiles per dim would give row=1 and col=2
    i = int(np.floor(x * tiles))

    return clip(i, 0, tiles - 1)

def getTilingIndex(dims: int, tiles_per_dim: int, pos: List, ):
    ind = 0

    total_tiles = tiles_per_dim ** dims

    for d in range(dims):
        # which cell am I in on this axis?
        axis = getAxisCell(pos[d], tiles_per_dim)
        already_seen = tiles_per_dim ** d
        ind += axis * already_seen

    # ensure we don't overflow into another tiling
    return clip(ind, 0, total_tiles - 1)

def getTCIndices(dims: int, tiles: int, tilings: int, offsets, pos):
    total_tiles = tiles**dims

    index = np.empty((tilings), dtype=np.int64)

    for ntl in range(tilings):
        ind = getTilingIndex(dims, tiles, pos + offsets[ntl])
        index[ntl] = ind + total_tiles * ntl

    return index

def minMaxScaling(x, mi, ma):
    return (x - mi) / (ma - mi)

def clip(x, mi, ma):
    return max(min(x, ma), mi)

# target function with gaussian noise
def target_fn(x, y):
    return np.sin(x) + np.cos(y) + 0.1 * np.random.randn()

tiles = 10
tilings = 8
dims = 2
input_ranges = np.asarray([
    (0, 2 * np.pi),
    (0, 2 * np.pi)
])

stepSize = 0.1 / tilings

offsets = np.array([ build_offset(dims, tiles, tilings, ntl) for ntl in range(tilings) ])

print(offsets)

total_tiles = tilings * tiles ** dims

weights = np.zeros(total_tiles)

def learn(x, y, z):
    indices = get_indices(dims, tiles, tilings, offsets, (x, y), input_ranges)

    estimate = 0

    for tile in indices:
        estimate += weights[tile]                  #form estimate

    error = z - estimate

    for tile in indices:
        weights[tile] += stepSize * error          #learn weights

def test(x, y):
    indices = get_indices(dims, tiles, tilings, offsets, (x, y), input_ranges)
    estimate = 0

    for tile in indices:
        estimate += weights[int(tile)]

    return estimate

# learn from 10,000 samples
for i in range(1000):
    # get noisy sample from target function at random location
    x, y = 2 * np.pi * np.random.rand(2)
    target = target_fn(x, y)
    
    learn(x, y, target)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# resolution
res = 200

# (x, y) space to evaluate
x = np.arange(0.0, 2.0 * np.pi, 2.0 * np.pi / res)
y = np.arange(0.0, 2.0 * np.pi, 2.0 * np.pi / res)

# map the function across the above space
z = np.zeros([len(x), len(y)])
for i in range(len(x)):
  for j in range(len(y)):
    z[i, j] = test(x[i], y[j])

# plot function
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
X, Y = np.meshgrid(x, y)
surf = ax.plot_surface(X, Y, z, cmap=plt.get_cmap('hot'))
plt.show()