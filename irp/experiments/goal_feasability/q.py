import hashlib
import os
from typing import Optional, Union
import gym
import random
import numpy as np
import matplotlib.pyplot as plt
from irp.experiments.goal_feasability.env import Env
from gym.wrappers import TimeLimit
from irp.wrappers import ExpandDims, Discretize
import irp.utils
from irp import envs, ROOT_DIR
import json
import irp.q

def make_test_env(env):
    test_subimages, test_sublabels = irp.utils.get_subimages('case10_10.png')
    test_subimage, test_sublabel = test_subimages[184], test_sublabels[184]

    test_env = Env(test_subimage, test_sublabel, 15)
    test_env = TimeLimit(test_env, 15)
    test_env = Discretize(test_env, [0, 0, 0], [1, 1, 1], env.observation_space.nvec)
    test_env._state_bins = env._state_bins

    return test_env

def evaluate(test_env: Env, qtable, render=False, eps=10):
    dissims = []

    for _ in range(eps):
        state = test_env.reset(threshold_i=14)
        state = tuple(state)
        done = False

        # Until the agent gets stuck in a hole or reaches the goal, keep training it
        while not done:
            action = np.argmax(qtable[state])
                
            # Implement this action and move the agent in the desired direction
            new_state, reward, _done, info = test_env.step(action)
            new_state = tuple(new_state)
            done = "TimeLimit.truncated" in info

            # Update our current state
            state = new_state

            if render:
                test_env.render()

        dissims.append(info['dissim'])

    return sum(dissims) / eps

def learn(env, episodes, alpha, gamma, epsilon, epsilon_decay, min_eps, learn_delay=0, write_log: Optional[Union[irp.q.Q, bool]] = True):
    model = None

    if isinstance(write_log, irp.q.Q):
        model = write_log
    elif write_log is True:
        model = irp.q.Q(env, 0.0, tensorboard_log=os.path.join(ROOT_DIR, 'results/goal_feasability'))
        model.learn(0)

    qtable = np.zeros(tuple(env.observation_space.nvec) + (3,))

    t = 0
    rewards = []
    ep_len = []
    dissims = []

    for e in range(episodes):
        state = tuple(env.reset(threshold_i=None))
        done = False
        t_old = t
        dissims_ = []

        # Until the agent gets stuck in a hole or reaches the goal, keep training it
        while not done:
            rnd = np.random.random()

            if rnd < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(qtable[state])
                
            # Implement this action and move the agent in the desired direction
            new_state, reward, done, info = env.step(action)
            new_state = tuple(new_state)

            rewards.append(reward)
            dissims_.append(info['dissim'])

            # if t % 100 == 0 and t > 0:
            #     model._tb_write("rollout//reward", np.mean(rewards[-100:]), t)
            #     model._tb_write("rollout//ep_len", np.mean(ep_len[-100:]), t)
            #     model._tb_write("rollout//dissim", np.mean(dissims[-100:]), t)
            #     model._tb_write("rollout//epsilon", epsilon, t)

            t += 1

            # Update Q(s,a)
            qtable[state][action] = qtable[state][action] + \
                                    alpha * (reward + gamma * np.max(qtable[new_state]) - qtable[state][action])
            
            # Update our current state
            state = new_state


        dissims.append(info['dissim'])
        ep_len.append(t - t_old)

        if e % 10 == 0 and write_log:
            model._tb_write("rollout//reward", np.mean(rewards[-100:]), e)
            model._tb_write("rollout//ep_len", np.mean(ep_len[-100:]), e)
            model._tb_write("rollout//dissim", np.mean(dissims[-100:]), e)
            model._tb_write("rollout//epsilon", epsilon, e)


        if e % 100 == 0 and write_log:
            # avg = evaluate(test_env, qtable)
            avg = 1
            model._tb_write("eval//dissim", avg, t)

        if e >= learn_delay:
            epsilon = max(epsilon - epsilon_decay, min_eps)

    return qtable

if __name__ == "__main__":
    subimages, sublabels = irp.utils.get_subimages('case10_11.png')
    subimage, sublabel = subimages[184], sublabels[184]

    env = Env(subimage, sublabel, 15)
    # env = TimeLimit(env, 15)
    bins = (140, 140, 140)
    env = Discretize(env, [0, 0, 0], [1, 1, 139], bins)
    test_env = make_test_env(env)

    # Hyperparameters
    episodes = 1000        # Total number of episodes
    alpha = 0.5            # Learning rate
    gamma = 0.9            # Discount factor
    epsilon = 1.0          # Amount of randomness in the action selection
    epsilon_decay = 0.01  # Fixed amount to decrease

    qtable = learn(
        env, episodes, alpha, gamma, epsilon, epsilon_decay
    )

    evaluate(test_env, qtable, eps=2, render=True)

    # for _ in range(2):
    #     state = env.reset(threshold_i=14)
    #     done = False

    #     # Until the agent gets stuck in a hole or reaches the goal, keep training it
    #     while True:
    #         action = np.argmax(qtable[state])
                
    #         # Implement this action and move the agent in the desired direction
    #         new_state, reward, done, info = env.step(action)
            
    #         # Update our current state
    #         state = new_state

    #         print(reward, done, info)
    #         env.render()

    #     print('done')

    # print('Q-table after training:')
    # print(json.dumps(qtable, indent=4))