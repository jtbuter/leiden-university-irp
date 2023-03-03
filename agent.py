import gymnasium as gym
import numpy as np
#Loading and rendering the gym environment

env = gym.make("Taxi-v3").env
env.reset()
env.render()

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

Q = np.zeros((env.observation_space.n, env.action_space.n))

