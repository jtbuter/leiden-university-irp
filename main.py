import numpy as np
import gymnasium as gym
import cv2
import matplotlib.pyplot as plt
from env import Trus, compute_dissimilarity, threshold_subimage, morph_mask

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

state_grid = create_uniform_grid([*low.values()], [*high.values()], [bins] * len(low))
state_size = tuple(len(splits) + 1 for splits in state_grid)

# subimage, sublabel, coord = next(zip(subimages, sublabels, coords))
subimage, sublabel, coord = subimages[88], sublabels[88], coords[88]
features, bins = state_grid.shape

alpha = 0.8
discount_factor = 0.99
epsilon = 0.95

train_episodes = 1
test_episodes = 100          
max_steps = 100

env = Trus(subimage, sublabel, state_grid)
state, info = env.reset(seed = 123, options = {})

Q = np.zeros((35, 35, 35, 15, 3), dtype = np.float32)

training_rewards = []  
epsilons = []

for episode in range(train_episodes):
    state, info = env.reset(seed = 123, options = {})
    total_training_rewards = []
    steps = 0
    
    while True:
        exp_exp_tradeoff = np.random.uniform(0, 1)

        if exp_exp_tradeoff > epsilon:
            action = tuple(np.unravel_index(Q[state].argmax(), Q[state].shape))
        else:
            action = tuple(env.action_space.sample())

        new_state, reward, done, truncated, info = env.step(action)

        sa_slice = state + action
        Q_old = Q[sa_slice]
        target = reward + discount_factor * np.max(Q[new_state])

        Q[sa_slice] = Q_old + alpha * (target - Q_old) 

        total_training_rewards.append(reward)      
        state = new_state         
        
        thresh_index, size_index = action
        threshold, size = env.actions[0][thresh_index], env.actions[1][size_index]

        mask = threshold_subimage(subimage, threshold)
        mask = morph_mask(mask, size)

        if done == True:
            break

        steps += 1

        print(np.mean(total_training_rewards[-100:]), compute_dissimilarity(mask, sublabel))

        print('did step')
    
    training_rewards.append(np.sum(total_training_rewards))
    epsilons.append(epsilon)
    
    

print ("Training score over time: " + str(sum(training_rewards) / train_episodes))