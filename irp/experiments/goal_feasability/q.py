import hashlib
import gym
import random
import numpy as np
import matplotlib.pyplot as plt
from env import Env
from gym.wrappers import TimeLimit
import irp.utils
from irp import envs
import json

subimages, sublabels = irp.utils.get_subimages('case10_11.png')
subimage, sublabel = subimages[184], sublabels[184]

env = Env(subimage, sublabel, 15)
obs = set([
        int(hashlib.sha256(
            str(envs.utils.apply_threshold(subimage, ti).flatten().tolist()).encode('utf-8')
        ).hexdigest(), 16) % 10**8 for ti in env.intensities
])
qtable = {bit_mask: [0] * 3 for bit_mask in obs}

# Hyperparameters
episodes = 1000        # Total number of episodes
alpha = 0.5            # Learning rate
gamma = 0.9            # Discount factor
epsilon = 1.0          # Amount of randomness in the action selection
epsilon_decay = 0.001  # Fixed amount to decrease

outcomes = []

for _ in range(episodes):
    state = env.reset()
    done = False

    # Until the agent gets stuck in a hole or reaches the goal, keep training it
    while not done:
        rnd = np.random.random()

        if rnd < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(qtable[state])
             
        # Implement this action and move the agent in the desired direction
        new_state, reward, done, info = env.step(action)

        print(reward, done, info, epsilon)

        # Update Q(s,a)
        qtable[state][action] = qtable[state][action] + \
                                alpha * (reward + gamma * np.max(qtable[new_state]) - qtable[state][action])
        
        # Update our current state
        state = new_state

    epsilon = max(epsilon - epsilon_decay, 0.05)

for _ in range(2):
    state = env.reset()
    done = False

    # Until the agent gets stuck in a hole or reaches the goal, keep training it
    while not done:
        action = np.argmax(qtable[state])
             
        # Implement this action and move the agent in the desired direction
        new_state, reward, done, info = env.step(action)
        
        # Update our current state
        state = new_state

        print(reward, done, info)
        env.render()

    print('done')

print('Q-table after training:')
print(json.dumps(qtable, indent=4))