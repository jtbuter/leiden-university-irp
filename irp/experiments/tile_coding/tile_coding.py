import irp.wrappers as wrappers

def hashcoords(coordinates, m, readonly=False):
    if isinstance(m, wrappers.utils.IHT): return m.getindex(tuple(coordinates), readonly)
    if isinstance(m, int): return hash(tuple(coordinates)) % m
    if m is None: return coordinates

from math import floor

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

if __name__ == "__main__":
    maxSize = 2048
    iht = wrappers.utils.IHT(maxSize)
    # weights = [0]*maxSize
    numTilings = 16
    stepSize = 0.1/numTilings

    import numpy as np

    # target function with gaussian noise
    def target_fn(x, y):
        return np.sin(x) + np.cos(y) + 0.1 * np.random.randn()

    def mytiles(x, y):
        scaleFactor = 10 / (2 * np.pi)
        return tiles(iht, numTilings, [x*scaleFactor,y*scaleFactor])

    def learn(x, y, z):
        tiles = mytiles(x, y)
        estimate = 0.0

        for tile in tiles:
            estimate += weights[tile]                  #form estimate

        error = z - estimate
        
        for tile in tiles:
            weights[tile] += stepSize * error          #learn weights

    def test(x, y):
        tiles = mytiles(x, y)
        
        estimate = 0.0

        for tile in tiles:
            estimate += weights[tile]
        return estimate 

    weights = [0]*maxSize

    # learn from 10,000 samples
    for i in range(1000):

        # get noisy sample from target function at random location
        x, y = 2.0 * np.pi * np.random.rand(2)
        target = target_fn(x, y)
        
        learn(x, y, target)

    x, y = 2.5, 3.1
    
    print((np.sin(x) + np.cos(y)), test(x, y))


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