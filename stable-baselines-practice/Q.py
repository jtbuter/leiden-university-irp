import traceback
import warnings
import time
import sys
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

from copy import deepcopy

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

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": BasePolicy
    }

    def __init__(
        self, env: Union[GymEnv, str], learning_rate: Union[float, Schedule],
        gamma: float = 0.99, tensorboard_log: Optional[str] = None,
        verbose: int = 0, monitor_wrapper: bool = True, seed: Optional[int] = None,
        supported_action_spaces: Optional[Tuple[spaces.Space, ...]] = None,
    ):
        super().__init__(
            policy='MlpPolicy', env=env, learning_rate=learning_rate, tensorboard_log=tensorboard_log,
            verbose=verbose, support_multi_env=False, monitor_wrapper=monitor_wrapper,
            seed=seed, supported_action_spaces=(spaces.Discrete,)
        )

        self.gamma = gamma

        self._setup_model()

    def _setup_model(self) -> None:
        self.q_table = np.zeros((self.observation_space.n, self.action_space.n))
        self.rollout = None

    def predict(self, observation, deterministic: bool = False):
        if not deterministic and np.random.uniform(0,1) < 0.2:
            action = self.action_space.sample()
        else:
            action = np.argmax(self.q_table[observation,:])

        return action

    def collect_rollouts(self, callback: BaseCallback, log_interval: Optional[int] = None) -> RolloutReturn:
        callback.on_rollout_start()
        continue_training = True

        action = self.predict(self._last_obs)
        next_state, reward, done, info = self.env.step([action])

        callback.update_locals(locals())

        if callback.on_step() is False:
            return RolloutReturn(1, 0, continue_training=False)

        self.num_timesteps += 1

        self._update_info_buffer(info, reward)
        self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

        current_state = deepcopy(self._last_obs)
        next_state = next_state.item()
        reward = reward.item()
        done = done.item()
        info = info[0]
        new_state = next_state

        if done:
            new_state = info['terminal_observation']
            self._episode_num += 1

            if log_interval is not None and self._episode_num % log_interval == 0:
                self._dump_logs()

        self.rollout = current_state, action, reward, new_state, done
        self._last_obs = next_state

        return RolloutReturn(1, int(done), continue_training)

    def train(self):
        current_state, action, reward, next_state, done = self.rollout

        q_value = (1-self.learning_rate) * self.q_table[current_state, action] + self.learning_rate*(reward + self.gamma*max(self.q_table[next_state,:]))
        
        self.q_table[current_state, action] = q_value

    def learn(
        self, total_timesteps: int, callback: MaybeCallback = None, log_interval: int = 1,
        tb_log_name: str = "run", reset_num_timesteps: bool = True, progress_bar: bool = False,
    ):
        total_timesteps, callback = self._setup_learn(
            total_timesteps, callback, reset_num_timesteps,
            tb_log_name, progress_bar
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(callback, log_interval)
        
            if rollout.continue_training is False:
                break

            self.train()

        callback.on_training_end()

        return self

    def _setup_learn(
        self, total_timesteps: int, callback: MaybeCallback = None,
        reset_num_timesteps: bool = True, tb_log_name: str = "run",
        progress_bar: bool = False
    ) -> Tuple[int, BaseCallback]:
        total_timesteps, callback = super()._setup_learn(
            total_timesteps, callback, reset_num_timesteps, tb_log_name, progress_bar
        )

        self._last_obs = self._last_obs.item()

        return total_timesteps, callback

    def _dump_logs(self) -> None:
        """
        Write log.
        """
        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time/episodes", self._episode_num, exclude="tensorboard")

        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")

        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))

        # Pass the number of timesteps for tensorboard
        self.logger.dump(step=self.num_timesteps)


class CustomCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

        self.rewards = []
        self.total_episode_reward = 0

    def _on_step(self) -> bool:
        reward, done = self.locals['reward'].item(), self.locals['done'].item()

        self.total_episode_reward += reward

        if done:
            self.rewards.append(self.total_episode_reward)
            self.total_episode_reward = 0

        if self.n_calls % 1000 == 0:
            print(self.n_calls, np.mean(self.rewards[-100:]))

callback = CustomCallback()
env = gym.make('FrozenLake-v1')
env = TimeLimit(env, 100)
model = Q(env, learning_rate=0.1, gamma=0.99, tensorboard_log="logs")

model.learn(100000, callback=callback, log_interval=1)

print(model.q_table)