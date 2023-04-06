import time
import sys
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

from copy import deepcopy

import numpy as np
import torch as th
from gym import spaces
from torch.nn import functional as F


from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import get_linear_fn, get_parameters_by_name, polyak_update, safe_mean
from stable_baselines3.common.vec_env import VecEnv

from irp import utils

class Q(BaseAlgorithm):

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": BasePolicy
    }

    def __init__(
        self, env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        gamma: float = 0.99,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0, monitor_wrapper: bool = True, seed: Optional[int] = None,
        supported_action_spaces: Optional[Tuple[spaces.Space, ...]] = None,
        use_sb3_env = True,
    ):
        super().__init__(
            policy='MlpPolicy', env=env, learning_rate=learning_rate, tensorboard_log=tensorboard_log,
            verbose=verbose, support_multi_env=False, monitor_wrapper=monitor_wrapper,
            seed=seed, supported_action_spaces=(spaces.Discrete,)
        )

        assert len(env.observation_space.shape) > 0, "State space too small, wrap in an ExpandDimsWrapper"

        self.gamma = gamma
        self.exploration_rate = 0.0
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.use_sb3_env = use_sb3_env
        
        self._setup_model()

    def _setup_model(self) -> None:
        dims = utils.get_dims(self.observation_space, self.action_space)

        self.q_table = np.zeros(dims)
        self.rollout = None
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )

    def predict(self, observation, deterministic: bool = False):
        observation = tuple(observation)

        # Perform ε-greedy action selection
        if not deterministic and np.random.rand() < self.exploration_rate:
            action = self.action_space.sample()
        else:
            action = np.argmax(self.q_table[observation])

        return action

    def _on_step(self):
        # Get the exploration rate corresponding to the current timestep
        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        self.logger.record("rollout/exploration_rate", self.exploration_rate)

    def collect_rollouts(self, callback: BaseCallback, log_interval: Optional[int] = None) -> RolloutReturn:
        callback.on_rollout_start()
        continue_training = True

        action = self.predict(self._last_obs)

        if self.use_sb3_env:
            action = [action]

        next_state, reward, done, info = self.env.step(action)

        callback.update_locals(locals())

        # Check if our callback makes us want to stop early
        if callback.on_step() is False:
            return RolloutReturn(1, 0, continue_training=False)

        self.num_timesteps += 1

        self._update_info_buffer(info, reward)
        self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

        self._on_step()

        # TODO: wrap deze code in een self-explanatory function?
        current_state = deepcopy(self._last_obs)
        next_state, reward, done, info = self._unwrap_sb3_env(next_state, reward, done, info)
        new_state = next_state

        # VenEnv resets the environment when reaching a terminal state or truncating
        # episodes, and returns this as the next state instead of the terminal state.
        # However, the real terminal state is made accessible through info.
        if done:
            new_state = info['terminal_observation']
            self._episode_num += 1

            if log_interval is not None and self._episode_num % log_interval == 0:
                self._dump_logs()

        # Current state moet altijd de next state uit self.action() worden,
        # maar de next state voor het updaten van q-values is afhankelijk van
        # of de episode beeindigd is.
        self.rollout = current_state, action, reward, new_state, done
        self._last_obs = next_state

        return RolloutReturn(1, int(done), continue_training)


    def _unwrap_sb3_env(self, *args):
        return tuple(arg[0] for arg in args)

    def train(self):
        current_state, action, reward, next_state, done = self.rollout

        # Cast states to tuples so they can be used for array indexing
        current_state = tuple(current_state)
        next_state = tuple(next_state)

        # Calculate the new q-values
        q_old = self.q_table[current_state][action]
        target = reward + self.gamma * max(self.q_table[next_state])
        q_new = q_old + self.learning_rate * (target - q_old)

        # Update the Q-table
        self.q_table[current_state][action] = q_new

    def learn(
        self, total_timesteps: int, callback: MaybeCallback = None, log_interval: int = 1,
        tb_log_name: str = "run", reset_num_timesteps: bool = True, progress_bar: bool = False,
    ):
        # Resets the environment, sets-up callbacks and prepares tensorboard logging
        total_timesteps, callback = self._setup_learn(
            total_timesteps, callback, reset_num_timesteps,
            tb_log_name, progress_bar
        )

        # Make local and global scope variables accessible to the callback
        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            # Perform a step in the environment and collect rewards
            rollout = self.collect_rollouts(callback, log_interval)
        
            if rollout.continue_training is False:
                break

            # Train on the collected data
            self.train()

        callback.on_training_end()

        return self

    def _setup_learn(
        self, total_timesteps: int, callback: MaybeCallback = None,
        reset_num_timesteps: bool = True, tb_log_name: str = "run",
        progress_bar: bool = False
    ) -> Tuple[int, BaseCallback]:
        # Resets the environment, sets-up callbacks and prepares tensorboard logging
        total_timesteps, callback = super()._setup_learn(
            total_timesteps, callback, reset_num_timesteps, tb_log_name, progress_bar
        )

        # Always work with one environment, so we can extract the first observation
        self._last_obs = self._last_obs[0]

        return total_timesteps, callback

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