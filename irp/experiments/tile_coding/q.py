from typing import Dict, Optional
import gym
import numpy as np

import os
import irp
import irp.wrappers as wrappers
import irp.q
from irp.experiments.tile_coding.policy import TiledQTable

def learn(environment: gym.Env, parameters: Dict, log: Optional[bool] = False):
    if log:
        model = irp.q.Q(environment, 0.0, tensorboard_log=os.path.join(irp.ROOT_DIR, 'results/tile_coding'))
        model.learn(0)

    episodes = parameters['episodes']
    alpha = parameters['alpha']
    gamma = parameters['gamma']
    epsilon = parameters['epsilon']
    epsilon_decay = parameters['epsilon_decay']
    min_epsilon = parameters['min_epsilon']
    learning_delay = parameters['learning_delay']

    # We re-initialize the Q-table
    qtable = TiledQTable(environment)

    # List of outcomes to plot
    outcomes = []
    steps = []
    d_sims = []

    # Training
    for e in range(episodes):
        state = environment.reset()
        done = False
        step = 0

        # By default, we consider our outcome to be a failure
        outcomes.append(0)

        # Until the agent gets stuck in a hole or reaches the goal, keep training it
        while not done:
        
            # Generate a random number between 0 and 1
            rnd = np.random.random()

            # If random number < epsilon, take a random action
            if rnd < epsilon:
                action = environment.action_space.sample()
            # Else, take the action with the highest value in the current state
            else:
                action = np.argmax(qtable.state_values(state))

            # Implement this action and move the agent in the desired direction
            new_state, reward, done, info = environment.step(action)

            d_sims.append(info['d_sim'])

            # Compute the target
            value = max(qtable.state_values(new_state))
            # Terminal states should have no future reward; multiply with (not `done`)
            target = reward + gamma * value * (not done)

            # Update Q(s,a)
            qtable.update(state, action, target, alpha)

            # Update our current state
            state = new_state

            step += 1

            # If we have a reward, it means that our outcome is a success
            if reward:
                outcomes[-1] = 1
                steps.append(step)

        if e % 10 == 0 and e > 0 and log:
            model._tb_write('rollout//dissim', np.mean(outcomes[-10:]), e)
            model._tb_write('rollout//ep_len', np.mean(steps[-10:]), e)

        if e >= learning_delay:
            # Update epsilon
            epsilon = max(epsilon - epsilon_decay, min_epsilon)

    return qtable