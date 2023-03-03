import cv2, numpy as np, matplotlib.pyplot as plt
import gymnasium as gym

def create_uniform_grid(low, high, bins):
    grid = list(np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] for dim in range(len(bins)))

    return np.asarray(grid)


def discretize(sample, grid):
    return list(int(np.digitize(s, g)) for s, g in zip(sample, grid))  # apply along each dimension


width, height = 512, 512
r_sim, c_sim = 16, 16

low = {
    'area': 0.,
    'compactness': 0.,
    'x': 0.,
    'y': 0.,
    'objects': 0.
}
high = {
    'area': 1.,
    'compactness': 1.,
    'x': 480.,
    'y': 480.,
    'objects': np.ceil((width / c_sim) / 2) * np.ceil((height / r_sim) / 2)
}
bins = [5] * 5

grid = create_uniform_grid([*low.values()], [*high.values()], bins)

print(list(grid[0]))

# gym.spaces.MultiDiscrete()
