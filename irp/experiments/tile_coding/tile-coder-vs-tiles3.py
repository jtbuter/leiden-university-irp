import numpy as np
import irp.wrappers as wrappers
import matplotlib.pyplot as plt
import timeit
from PyFixedReps import TileCoder

# target function with gaussian noise
def target_fn(x, y):
    return np.sin(x) + np.cos(y) + 0.1 * np.random.randn()

def update(w, tiles, alpha, target, value):
    w[tiles] += alpha * (target - value)

def get(w, tiles):
    return w[tiles].mean()

def plot(w, tiles_lambda):
    # resolution
    res = 200

    # (x, y) space to evaluate
    x = np.arange(0.0, 2.0 * np.pi, 2.0 * np.pi / res)
    y = np.arange(0.0, 2.0 * np.pi, 2.0 * np.pi / res)

    # map the function across the above space
    z = np.zeros([len(x), len(y)])

    for i in range(len(x)):
        for j in range(len(y)):
            tiles = tiles_lambda(x[i], y[j])
            z[i, j] = get(w, tiles)

    # plot function
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, z, cmap=plt.get_cmap('hot'))
    plt.show()

def evaluate(w, tiles_lambda):
    # resolution
    res = 200

    # (x, y) space to evaluate
    x = np.arange(0.0, 2.0 * np.pi, 2.0 * np.pi / res)
    y = np.arange(0.0, 2.0 * np.pi, 2.0 * np.pi / res)

    # map the function across the above space
    z = np.zeros([len(x), len(y)])
    t = np.zeros([len(x), len(y)])

    for i in range(len(x)):
        for j in range(len(y)):
            tiles = tiles_lambda(x[i], y[j])
            z[i, j] = get(w, tiles)
            t[i, j] = np.sin(x[i]) + np.cos(y[j])

    z = z.flatten()
    t = t.flatten()

    return np.mean(abs(z - t)), np.std(abs(z - t))

def tilecoder(tiles_per_dim, value_limits, tilings, alpha, do_plot = False):
    T = wrappers.utils.TileCoder((tiles_per_dim,) * 2, [value_limits] * 2, tilings)

    # linear function weight vector
    w = np.zeros(T.n_tiles)

    # learn from 10,000 samples
    for i in range(10000):
        # get noisy sample from target function at random location
        x, y = 2.0 * np.pi * np.random.rand(2)
        target = target_fn(x, y)
        # get prediction from active tiles at that location
        tiles = T[x, y]
        value = get(w, tiles)
        # update weights with SGD
        update(w, tiles, alpha, target, value)

    if do_plot:
        plot(w, lambda x, y: T[x, y])

    return w, lambda x, y: T[x, y]

def tiles3(tiles_per_dim, value_limits, tilings, alpha, do_plot = False):
    size = 10000
    iht = wrappers.utils.IHT(size)

    # linear function weight vector
    w = np.zeros(size)
    scale = tiles_per_dim / (value_limits[1] - value_limits[0])

    # learn from 10,000 samples
    for i in range(10000):
        # get noisy sample from target function at random location
        x, y = 2.0 * np.pi * np.random.rand(2)
        target = target_fn(x, y)
        # get prediction from active tiles at that location

        tiles = wrappers.utils.tiles(iht, numtilings=tilings, floats=[x * scale, y * scale])
        value = get(w, tiles)
        # update weights with SGD
        update(w, tiles, alpha, target, value)

    if do_plot:
        plot(w, lambda x, y: wrappers.utils.tiles(iht, tilings, [x * scale, y * scale]))

    return w, lambda x, y: wrappers.utils.tiles(iht, tilings, [x * scale, y * scale])

def pyfixedreps(tiles_per_dim, value_limits, tilings, alpha, do_plot = False):
    tc = TileCoder({
        # [required]
        'tiles': tiles_per_dim, # how many tiles in each tiling
        'tilings': tilings,
        'dims': 2, # shape of the state-vector
        'input_ranges': [value_limits] * 2, # a vector of same length as 'dims' containing (min, max) tuples to rescale inputs
    })


    # linear function weight vector
    w = np.zeros(tc.features())

    # learn from 10,000 samples
    for i in range(10000):
        # get noisy sample from target function at random location
        x, y = 2.0 * np.pi * np.random.rand(2)
        target = target_fn(x, y)
        # get prediction from active tiles at that location
        tiles = tc.get_indices((x, y))
        value = get(w, tiles)
        # update weights with SGD
        update(w, tiles, alpha, target, value)

    if do_plot:
        plot(w, lambda x, y: tc.get_indices((x, y)))

    return w, lambda x, y: tc.get_indices((x, y))

tiles_per_dim = 2
value_limits = (0, 2 * np.pi)
tilings = 32
alpha = 0.1

w_tc, f_tc = tilecoder(tiles_per_dim, value_limits, tilings, alpha)
w_t3, f_t3 = tiles3(tiles_per_dim, value_limits, tilings, alpha)
w_tp, f_tp = pyfixedreps(tiles_per_dim, value_limits, tilings, alpha)

print('tc', evaluate(w_tc, f_tc))
print('t3', evaluate(w_t3, f_t3))
print('tp', evaluate(w_tp, f_tp))
# print('tilecoder', timeit.timeit(lambda: evaluate(w_tc, f_tc), number=10) / 10, 'seconds')
# print('tiles3', timeit.timeit(lambda: evaluate(w_t3, f_t3), number=10) / 10, 'seconds')
