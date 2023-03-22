from env import UltraSoundEnv
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO, DQN
import gymnasium as gym

def create_subimages(image, rows, cols):
    height, width = image.shape
    subimage_width, subimage_height = int(width / cols), int(height / rows)
    subimages, coords = [], []

    for y in range(0, height, subimage_height):
        for x in range(0, width, subimage_width):
            subimage = image[y:y + subimage_height, x:x + subimage_width]

            subimages.append(subimage)
            coords.append((x, y))

    return subimages, coords

image = cv.imread("../data/trus/images/case10_11.png", cv.IMREAD_GRAYSCALE)
label = cv.imread("../data/trus/labels/case10_11.png", cv.IMREAD_GRAYSCALE)
subimage = create_subimages(image, rows = 32, cols = 16)[0][184]
sublabel = create_subimages(label, rows = 32, cols = 16)[0][184]

env = UltraSoundEnv(sample = subimage, label = sublabel, n_thresholds = 20)

model = DQN(policy = "CnnPolicy", env = env, verbose = 1)
model.learn(total_timesteps = 25000)

obs = env.reset()

for i in range(1000):
    action, _state = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    
    env.render()

    if done:
        obs = env.reset()

        print('Done')


# print(env.dissim)

# while 1:
#     env.reset()
#     state, reward, done, _, _ = env.step(env.action_space.sample())

#     print(env.dissim)
#     print(env._compute_reward(sublabel))

#     plt.imshow(state, cmap = 'gray', vmin = 0, vmax = 1)
#     plt.show()
