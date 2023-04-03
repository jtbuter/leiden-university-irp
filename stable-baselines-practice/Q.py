import traceback
import warnings
import time
import sys
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
import gym
from gym import spaces
from gym.wrappers import TimeLimit
from torch.nn import functional as F

from stable_baselines3.common import utils
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import get_linear_fn, get_parameters_by_name, polyak_update, safe_mean
from stable_baselines3.common.vec_env import VecEnv
from QPolicy import QPolicy

class Q(BaseAlgorithm):
    def _setup_model(self) -> None:
        pass

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": BasePolicy
    }

    def __init__(self, env, learning_rate, gamma):
        super().__init__('MlpPolicy', env, learning_rate)

        # self.env = env

        n_observations = env.observation_space.n
        n_actions = env.action_space.n
        self.Q_table = np.zeros((n_observations, n_actions))

        self.max_n_timesteps = 100000
        self.max_iter_episode = 100
        self.exploration_proba = 1
        self.exploration_decreasing_decay = 0
        self.min_exploration_proba = 0.01
        self.gamma = gamma
        self.lr = learning_rate

        self.total_rewards_episode = []
        self.n_timesteps = 0
        self.num_episode = 0

        self.current_state = None
        self.action = None
        self.reward = None
        self.done = None
        self.info = None
        self.next_state = None
        self.callback = None
        self.total_episode_reward = 0

    def train(self):
        self.n_timesteps += 1

        if np.random.uniform(0,1) < 0.2:
            self.action = self.action_space.sample()
        else:
            self.action = np.argmax(self.Q_table[self.current_state,:])

        self.next_state, self.reward, self.done, self.info = self.env.step([self.action])
        self.next_state = self.next_state.item()
        self.reward = self.reward.item()
        self.done = self.done.item()
        self.info = self.info[0]

        if self.done:
            self.next_state = self.info['terminal_observation']

        q_value = (1-self.lr) * self.Q_table[self.current_state, self.action] + self.lr*(self.reward + self.gamma*max(self.Q_table[self.next_state,:]))
        self.Q_table[self.current_state, self.action] = q_value
        self.total_episode_reward += self.reward

        self.current_state = self.next_state

    def learn(self, total_timesteps, callback):
        self.callback = callback
        self.current_state = self.env.reset().item()

        timesteps = 0

        while self.n_timesteps < total_timesteps:
            self.train()
        
            callback._on_step()

            timesteps += 1

            # If the episode is finished, we leave the for loop
            if self.done:
                self.current_state = self.env.reset().item()
                self.callback.rewards.append(self.total_episode_reward)
                self.total_episode_reward = 0
                self.num_episode += 1
                timesteps = 0

class CustomCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

        self.rewards = []

    def _on_step(self) -> bool:
        self.n_calls += 1

        if self.n_calls % 1000 == 0:
            print(self.n_calls, np.mean(self.rewards[-100:]))

callback = CustomCallback()
env = gym.make('FrozenLake-v1')
env = TimeLimit(env, 100)
model = Q(env, learning_rate=0.1, gamma=0.99)

model.learn(100000, callback)

print(model.Q_table)