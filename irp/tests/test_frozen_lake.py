"""
Evaluating the performance of our Q-learning model on the FrozenLake-v1 gym
environment.
"""

import time
import gym
import os
import numpy as np

from irp import ROOT_DIR
from irp.wrappers import ExpandDims
from irp.q import Q

environment = gym.make("FrozenLake-v1", is_slippery=True)
environment = ExpandDims(environment)
environment.reset()

model = Q(
    environment, learning_rate=0.5, gamma=0.9, exploration_delay=0.0,
    tensorboard_log=os.path.join(ROOT_DIR, 'results/annealing')
)
model.learn(10000)


episodes = 100
nb_success = 0

# Evaluation
for _ in range(episodes):
    state = environment.reset()
    done = False
    
    # Until the agent gets stuck or reaches the goal, keep training it
    while not done:
        # Choose the action with the highest value in the current state
        action = np.argmax(model.policy.q_table[state])

        # Implement this action and move the agent in the desired direction
        new_state, reward, done, info = environment.step(action)

        # Update our current state
        state = new_state

        # When we get a reward, it means we solved the game
        nb_success += reward

# Let's check our success rate!
print (f"Success rate = {nb_success/episodes*100}%")