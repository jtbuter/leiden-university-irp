import numpy as np
import gymnasium as gym
import cv2
import matplotlib.pyplot as plt
from env import Trus

def create_subimages(image, rows, cols):
    height, width = image.shape
    subimage_width = width / cols
    subimage_height = height / rows
    
    assert int(subimage_height) == subimage_height and int(subimage_width) == subimage_width

    subimage_height = int(subimage_height)
    subimage_width = int(subimage_width)

    subimages = []
    coords = []

    for y in range(0, height, subimage_height):
        for x in range(0, width, subimage_width):
            subimage = image[y:y + subimage_height,x:x + subimage_width]

            subimages.append(subimage)
            coords.append((x, y))

    return subimages, coords


def create_uniform_grid(low, high, bins):
    grid = list(np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] for dim in range(len(bins)))

    return np.asarray(grid)


image = cv2.imread("../data/trus/images/case10_10.png", cv2.IMREAD_GRAYSCALE)
label = cv2.imread("../data/trus/labels/case10_10.png", cv2.IMREAD_GRAYSCALE)

subimages, coords = create_subimages(image, rows = 16, cols = 16)
sublabels, coords = create_subimages(label, rows = 16, cols = 16)

bins = 35
r_sim, c_sim = subimages[0].shape
low = {'area': 0., 'compactness': 0., 'objects': 0.}
high = {'area': 1.,'compactness': 1., 'objects': np.ceil(c_sim / 2) * np.ceil(r_sim / 2)}

state_grid = create_uniform_grid([*low.values()], [*high.values()], [bins] * 5)
state_size = tuple(len(splits) + 1 for splits in state_grid)
q_table = np.zeros((state_size + (1,)))

for subimage, sublabel, coord in zip(subimages, sublabels, coords):
    features, bins = state_grid.shape

    env = Trus(subimage, sublabel, *coord, state_grid)
    observation, info = env.reset(seed = 123, options={})

    break
    
#     for i in range(3):
#         action = env.action_space.sample()

#         print(env.step(action))
