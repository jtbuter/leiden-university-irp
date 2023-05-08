"""
Tile Coding Software version 3.0beta
by Rich Sutton
based on a program created by Steph Schaeffer and others
External documentation and recommendations on the use of this code is available in the
reinforcement learning textbook by Sutton and Barto, and on the web.
These need to be understood before this code is.

This software is for Python 3 or more.

This is an implementation of grid-style tile codings, based originally on
the UNH CMAC code (see http://www.ece.unh.edu/robots/cmac.htm), but by now highly changed.
Here we provide a function, "tiles", that maps floating and integer
variables to a list of tiles, and a second function "tiles-wrap" that does the same while
wrapping some floats to provided widths (the lower wrap value is always 0).

The float variables will be gridded at unit intervals, so generalization
will be by approximately 1 in each direction, and any scaling will have
to be done externally before calling tiles.

Num-tilings should be a power of 2, e.g., 16. To make the offsetting work properly, it should
also be greater than or equal to four times the number of floats.

The first argument is either an index hash table of a given size (created by (make-iht size)),
an integer "size" (range of the indices from 0), or nil (for testing, indicating that the tile
coordinates are to be returned without being converted to indices).
"""

basehash = hash

class IHT:
    "Structure to handle collisions"
    def __init__(self, sizeval):
        self.size = sizeval
        self.overfullCount = 0
        self.dictionary = {}

    def __str__(self):
        "Prepares a string for printing whenever this object is printed"
        return "Collision table:" + \
               " size:" + str(self.size) + \
               " overfullCount:" + str(self.overfullCount) + \
               " dictionary:" + str(len(self.dictionary)) + " items"

    def count (self):
        return len(self.dictionary)

    def fullp (self):
        return len(self.dictionary) >= self.size

    def getindex (self, obj, readonly=False):
        d = self.dictionary
        if obj in d: return d[obj]
        elif readonly: return None
        size = self.size
        count = self.count()
        if count >= size:
            if self.overfullCount==0: print('IHT full, starting to allow collisions')
            self.overfullCount += 1
            return basehash(obj) % self.size
        else:
            d[obj] = count
            return count

def hashcoords(coordinates, m, readonly=False):
    if isinstance(m, IHT): return m.getindex(tuple(coordinates), readonly)
    if isinstance(m, int): return basehash(tuple(coordinates)) % m
    if m is None: return coordinates

from math import floor, log
from itertools import zip_longest

def tiles(ihtORsize, numtilings, floats, ints=[], readonly=False):
    """returns num-tilings tile indices corresponding to the floats and ints"""
    qfloats = [floor(f*numtilings) for f in floats]
    Tiles = []
    for tiling in range(numtilings):
        tilingX2 = tiling*2
        coords = [tiling]
        b = tiling
        for q in qfloats:
            coords.append( (q + b) // numtilings )
            b += tilingX2
        coords.extend(ints)
        Tiles.append(hashcoords(coords, ihtORsize, readonly))
    return Tiles

maxSize = 2048
iht = IHT(maxSize)
weights = [0]*maxSize
numTilings = 8
stepSize = 0.1/numTilings

import numpy as np

# target function with gaussian noise
def target_fn(x, y):
    return np.sin(x) + np.cos(y) + 0.1 * np.random.randn()

def mytiles(x, y):
    scaleFactor = 10.0/(2 * np.pi)
    return tiles(iht, numTilings, [x*scaleFactor,y*scaleFactor])

def learn(x, y, z):
    tiles = mytiles(x, y)
    estimate = 0
    for tile in tiles:
        estimate += weights[tile]                  #form estimate
    error = z - estimate
    for tile in tiles:
        weights[tile] += stepSize * error          #learn weights

def test(x, y):
    tiles = mytiles(x, y)
    estimate = 0
    for tile in tiles:
        estimate += weights[tile]
    return estimate 

avg_error = []

for j in range(200):
    print(j)

    # learn from 10,000 samples
    for i in range(1000):
        # get noisy sample from target function at random location
        x, y = 2.0 * np.pi * np.random.rand(2)
        target = target_fn(x, y)
        
        learn(x, y, target)

    x, y = 2.5, 3.1

    avg_error.append(abs(test(x, y) - (np.sin(x) + np.cos(y))))

print(np.mean(avg_error))

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # resolution
# res = 200

# # (x, y) space to evaluate
# x = np.arange(0.0, 2.0 * np.pi, 2.0 * np.pi / res)
# y = np.arange(0.0, 2.0 * np.pi, 2.0 * np.pi / res)

# # map the function across the above space
# z = np.zeros([len(x), len(y)])
# for i in range(len(x)):
#   for j in range(len(y)):
#     z[i, j] = test(x[i], y[j])

# # plot function
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# X, Y = np.meshgrid(x, y)
# surf = ax.plot_surface(X, Y, z, cmap=plt.get_cmap('hot'))
# plt.show()