import traceback
import warnings
import time
import sys
from copy import deepcopy
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

SelfQ = TypeVar("SelfQ", bound="Q")

class Q(BaseAlgorithm):
    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": BasePolicy
    }

    def __init__(self, env, learning_rate):
        super().__init__('MlpPolicy', env, learning_rate)

        self.rollout = None
        self.Q_table = None
        self.gamma = 0.99

        self._setup_model()
        
    def _setup_model(self) -> None:
        self.Q_table = np.zeros((self.observation_space.n, self.action_space.n))

    def train(self):
        current_state, action, reward, next_state, done = self.rollout
        lr = self.learning_rate
        gamma = self.gamma

        q_new = (1-lr) * self.Q_table[current_state, action] +lr*(reward + gamma*max(self.Q_table[next_state,:]))

        self.Q_table[current_state, action] = q_new

    def learn(
        self: SelfQ, total_timesteps: int, callback: MaybeCallback = None, log_interval: int = 1,
        tb_log_name: str = "run", reset_num_timesteps: bool = True, progress_bar: bool = False
    ):
        total_timesteps, callback = self._setup_learn(
            total_timesteps, callback, reset_num_timesteps, tb_log_name, progress_bar
        )

        total_episode_reward = 0

        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollout(callback, log_interval)

            if rollout.continue_training is False:
                break

            self.train()

            reward, done = self.rollout[2], self.rollout[4]

            total_episode_reward += reward

            if done:
                callback.rewards.append(total_episode_reward)

                total_episode_reward = 0

            self.num_timesteps += 1

    def collect_rollout(
        self, callback: BaseCallback, log_interval: Optional[int] = None
    ) -> RolloutReturn:
        callback.on_rollout_start()
        continue_training = True

        current_state = deepcopy(self._last_obs)
        action = self._sample_action()
        next_state, reward, done, info = self._do_step(action)

        self.rollout = current_state, action, reward, next_state, done

        # Give access to local variables
        callback.update_locals(locals())

        # Only stop training if return value is False, not when it is None.
        if callback.on_step() is False:
            return RolloutReturn(1, 0, continue_training=False)

        self._last_obs = deepcopy(next_state)

        callback.on_rollout_end()

        return RolloutReturn(1, 0, continue_training)

    def _sample_action(self):
        if np.random.uniform(0, 1) < 0.2:
            action = self.action_space.sample()
        else:
            action = np.argmax(self.Q_table[self._last_obs,:])

        return action

    def _do_step(self, action):
        current_state = self._last_obs
        new_obs, rewards, dones, infos = self.env.step([action])
        new_ob, reward, done, info = new_obs[0], rewards[0], dones[0], infos[0]

        if done:
            new_ob = info['terminal_observation']

        return new_ob, reward, done, info

    def _setup_learn(
        self, total_timesteps: int, callback: MaybeCallback = None, reset_num_timesteps: bool = True,
        tb_log_name: str = "run", progress_bar: bool = False
    ) -> Tuple[int, BaseCallback]:
        _learn = super()._setup_learn(
            total_timesteps, callback, reset_num_timesteps, tb_log_name, progress_bar
        )

        self._last_obs = self._last_obs.item()

        return _learn

class CustomCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

        self.n_calls = 0
        self.rewards = []

    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:
            print(self.n_calls, np.mean(self.rewards[-100:]))

        return True

callback = CustomCallback()
env = gym.make('FrozenLake-v1')
env = TimeLimit(env, 100)
model = Q(env, learning_rate=0.1)
model.learn(total_timesteps=100000, callback=callback)
print(model.Q_table)