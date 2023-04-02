import traceback
import warnings
import time
import sys
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gym import spaces
from gym.wrappers import TimeLimit
from torch.nn import functional as F

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import get_linear_fn, get_parameters_by_name, polyak_update, safe_mean
from stable_baselines3.common.vec_env import VecEnv
from QPolicy import QPolicy

Q = TypeVar("Q", bound="Q")
MlpPolicy = QPolicy

class Q(BaseAlgorithm):
    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": MlpPolicy,
    }

    def __init__(self,
        env: Union[GymEnv, str], learning_rate: Union[float, Schedule] = 1e-4, gamma: float = 0.99,
        exploration_fraction: float = 0.1, exploration_initial_eps: float = 1.0, exploration_final_eps: float = 0.05,
        tensorboard_log: Optional[str] = None, verbose: int = 0, seed: Optional[int] = None, _init_setup_model: bool = True
    ) -> None:
        super().__init__('MlpPolicy', env, learning_rate, tensorboard_log=tensorboard_log, verbose=verbose, seed=seed, supported_action_spaces=(spaces.Discrete,))

        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.gamma = gamma

        self._last_actions = None # type: np.ndarray
        self._last_rewards = None # type: np.ndarray
        self._next_obs = None # type: np.ndarray

        self.q_table = np.zeros((16, 4))

        # For updating the target network with multiple envs:
        self._n_calls = 0
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = 0.0

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )

    def _on_step(self) -> None:
        self._n_calls += 1

        self.exploration_rate = 0.2

        self.logger.record("rollout/exploration_rate", self.exploration_rate)

    def _get_sars(self):
        state = self._last_obs.item()
        action = self._last_actions.item()
        reward = self._last_rewards.item()
        next_state = self._next_obs.item()

        return state, action, reward, next_state

    def train(self) -> None:
        current_state, action, reward, next_state = self._get_sars()
        lr = self.learning_rate
        gamma = self.gamma
        q_new = (1-lr)*self.q_table[current_state, action] + lr*(reward + gamma*max(self.q_table[next_state,:]))

        # print(q_new)

        # print(self.q_table)

        self.q_table[current_state, action] = q_new
        # q_old = self.policy.q_table[s, a]
        # target = r + self.gamma * np.max(self.policy.q_table[sp])
        # q_new = q_old + self.learning_rate * (target - q_old)

        # self.policy.q_table[s, a] = q_new

        # print(s, a, r, sp)

    def learn(self: Q,
        total_timesteps: int, callback: MaybeCallback = None, log_interval: int = 4, tb_log_name: str = "Q",
        reset_num_timesteps: bool = True, progress_bar: bool = False
    ) -> Q:
        total_timesteps, callback = self._setup_learn(
            total_timesteps, callback, reset_num_timesteps, tb_log_name, progress_bar
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(self.env, callback=callback, log_interval=log_interval)

            if rollout.continue_training is False:
                break

            self.train()

        callback.on_training_end()

        return self


    def predict(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """

        if np.random.uniform(0,1) < 0.2:
            action = np.array([self.action_space.sample()])
        else:
            action = np.argmax(self.q_table[observation,:])

        # if not deterministic and np.random.rand() < self.exploration_rate:
        #     action = np.array([self.action_space.sample()])
        # else:
        #     action = self.policy.predict(observation, deterministic)

        return action.reshape(-1)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        assert isinstance(env, VecEnv), "You must pass a VecEnv"

        callback.on_rollout_start()
        continue_training = True

        # Select action randomly or according to policy
        actions = self.predict(self._last_obs, deterministic=False)

        # Perform action
        new_obs, rewards, dones, infos = env.step(actions)

        self._last_actions = actions
        self._last_rewards = rewards
        self._next_obs = new_obs

        self.num_timesteps += 1

        # Give access to local variables
        callback.update_locals(locals())

        # Only stop training if return value is False, not when it is None.
        if callback.on_step() is False:
            return RolloutReturn(0, 1, continue_training=False)

        # Retrieve reward and episode length if using Monitor wrapper
        self._update_info_buffer(infos, dones)
        self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)
        self._on_step()

        done = dones.item()

        if done:
            self._episode_num += 1

            # Log training infos
            if log_interval is not None and self._episode_num % log_interval == 0:
                self._dump_logs()

        callback.on_rollout_end()

        return RolloutReturn(1, 1, continue_training)         

    def _dump_logs(self) -> None:
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

import gym

class CustomCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

        self.episode_rewards = []
        self.total_rewards = []

    def _on_step(self) -> bool:
        reward = self.locals['rewards'].item()
        done = self.locals['dones'].item()

        self.episode_rewards.append(reward)

        if done:
            self.total_rewards.append(sum(self.episode_rewards))
            self.episode_rewards = []

        if self.n_calls % 100 == 0 and self.n_calls > 0:
            print(self.num_timesteps, np.mean(self.total_rewards[-100:]))

callback = CustomCallback()
env = gym.make('FrozenLake-v1')
env = TimeLimit(env, 100)
model = Q(env, learning_rate=0.1)

model.learn(100000, callback=callback)