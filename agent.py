import gymnasium as gym
import numpy as np
from env import Trus
#Loading and rendering the gym environment

env = Trus()
env.reset()

#Setting the hyperparameters
alpha = 0.7 #learning rate                 
discount_factor = 0.618               
epsilon = 1                  
max_epsilon = 1
min_epsilon = 0.01         
decay = 0.01

train_episodes = 2000    
test_episodes = 100          
max_steps = 100

print(env.observation_space.nvec)

Q = np.zeros(env.observation_space.n + env.action_space.n)

