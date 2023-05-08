import numpy as np

from typing import List

import irp.wrappers as wrappers

def create_tiling(feat_range, bins, offset):
    """
    Create 1 tiling spec of 1 dimension(feature)
    feat_range: feature range; example: [-1, 1]
    bins: number of bins for that feature; example: 10
    offset: offset for that feature; example: 0.2
    """
    
    return np.linspace(feat_range[0], feat_range[1], bins+1)[1:-1] + offset
  
# def create_tilings(feature_ranges, number_tilings, bins, offsets):
#         """
#         feature_ranges: range of each feature; example: x: [-1, 1], y: [2, 5] -> [[-1, 1], [2, 5]]
#         number_tilings: number of tilings; example: 3 tilings
#         bins: bin size for each tiling and dimension; example: [[10, 10], [10, 10], [10, 10]]: 3 tilings * [x_bin, y_bin]
#         offsets: offset for each tiling and dimension; example: [[0, 0], [0.2, 1], [0.4, 1.5]]: 3 tilings * [x_offset, y_offset]
#         """
#         tilings = []
#         # for each tiling
#         for tile_i in range(number_tilings):
#             tiling_bin = bins[tile_i]
#             tiling_offset = offsets[tile_i]

#             tiling = []
#             # for each feature dimension
#             for feat_i in range(len(feature_ranges)):
#                 feat_range = feature_ranges[feat_i]
#                 # tiling for 1 feature
#                 feat_tiling = create_tiling(feat_range, tiling_bin[feat_i], tiling_offset[feat_i])
#                 tiling.append(feat_tiling)
#             tilings.append(tiling)
#         return np.array(tilings)

# feature_ranges = [[-1, 1], [2, 5]]  # 2 features
# number_tilings = 3
# bins = [[10, 10], [10, 10], [10, 10]]  # each tiling has a 10*10 grid
# offsets = [[0, 0], [0.2, 1], [0.4, 1.5]]

# tilings = create_tilings(feature_ranges, number_tilings, bins, offsets)

# print(tilings.shape)  # # of tilings X features X bins

def getTileIndex(dimensions: int, tiles_per_dim: int, coords: List[float]) -> int:
    # the index of the tile that coords is in
    ind = 0

    # length of each tile in each dimension
    # if there are 2 tiles, then each is length 1/2
    # if there are 3 tiles, then each is length 1/3
    # ...
    tile_length = 1 / tiles_per_dim

    # the total number of tiles in a 1D space
    # is exactly the number of requested tiles.
    # in 2D space, we have a square so we square the number of tiles
    # in 3D space, we have a cube so we cube the number of tiles
    # ...
    total_tiles = tiles_per_dim ** dimensions

    for dim in range(dimensions):
        # in this particular dimension, find out how
        # far from 0 we are and how close to 1
        ind += coords[dim] // tile_length

        # then offset that index by the number of tiles
        # in all dimensions before this one
        ind *= tiles_per_dim**dim

    # make sure the tile index does not go outside the
    # range of the total number of tiles in this tiling
    # this causes the index to wrap around in a hyper-sphere
    # when the inputs are not between [0, 1]
    return ind % total_tiles

def getTilingsIndices(dimensions: int, tiles: int, tilings: int, coords: List[float]) -> List[int]:
    tiles_per_tiling = tiles**dimensions
    tile_length = 1 / tiles

    indices = np.empty(tilings)

    for tiling in range(tilings):
        # offset each tiling by a fixed percent of a tile-length
        # first tiling is not offset
        # second is offset by 1 / tilings percent
        # third is offset by 2 / tilings percent
        # ...
        offset = tile_length * tiling / tilings

        # because this wraps around when the inputs are
        # bigger than 1, this is a safe operation
        ind = getTileIndex(dimensions, tiles, [coord + offset for coord in coords])

        # store the index, but first offset it by the number
        # of tiles in all tilings before us
        indices[tiling] = ind + tiles_per_tiling * tiling

    return indices

for x in range(10):
    x /= 10

    for y in range(1):
        y /= 10

        print(getTilingsIndices(2, 10, 2, [x, y]))

# grid = [create_tiling([0, 1], 100, 0)]
# grid = wrappers.Discretize.make_state_bins((2,), (0,), (1,))

# print(grid)

# print(wrappers.utils.discrete([1], grid))