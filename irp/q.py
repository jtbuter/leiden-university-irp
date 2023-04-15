import io
import os
import glob
import pathlib
import time
import sys
from typing import Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union

from copy import deepcopy

from gym import spaces
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule
from stable_baselines3.common.utils import get_linear_fn, safe_mean
from stable_baselines3.common.logger import TensorBoardOutputFormat

import irp.utils
from irp.policies import QPolicy

SelfQ = TypeVar("SelfQ", bound="Q")

class Q(BaseAlgorithm):
    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": QPolicy
    }

    def __init__(
        self,
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        gamma: float = 0.99,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0, monitor_wrapper: bool = True,
        seed: Optional[int] = None,
        use_sb3_env = True,
        init_setup_model=True
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

        self.rollout = None

        if init_setup_model:
            self._setup_model()

    def _setup_model(self: SelfQ) -> None:
        self.set_random_seed(self.seed)

        self.policy = QPolicy(self.observation_space, self.action_space)
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )

    def predict(self: SelfQ, observation, deterministic: bool = False):
        # Perform ε-greedy action selection
        if not deterministic and np.random.rand() < self.exploration_rate:
            action = self.action_space.sample()
        else:
            action = self.policy.predict(observation=observation)

        return action

    def _on_step(self: SelfQ):
        # Get the exploration rate corresponding to the current timestep
        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        self.logger.record("rollout/exploration_rate", self.exploration_rate)

    def collect_rollouts(
        self, callback: BaseCallback, log_interval: Optional[int] = None,
        log_every_timestep: Optional[bool] = False
    ) -> RolloutReturn:
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
        next_state, reward, done, info = irp.utils.unwrap_sb3_env(next_state, reward, done, info)
        new_state = next_state

        # VenEnv resets the environment when reaching a terminal state or truncating
        # episodes, and returns this as the next state instead of the terminal state.
        # However, the real terminal state is made accessible through info.
        if done:
            new_state = info['terminal_observation']
            self._episode_num += 1

            # Logs every timestep or once every n episodes
            if (log_interval is not None and self._episode_num % log_interval == 0) or log_every_timestep:
                self._dump_logs()

        # Current state moet altijd de next state uit self.action() worden,
        # maar de next state voor het updaten van q-values is afhankelijk van
        # of de episode beeindigd is.
        self.rollout = current_state, action, reward, new_state, done
        self._last_obs = next_state

        return RolloutReturn(1, int(done), continue_training)

    def train(self: SelfQ):
        current_state, action, reward, next_state, done = self.rollout

        # Cast states to tuples so they can be used for array indexing
        current_state = tuple(current_state)
        next_state = tuple(next_state)

        q_old = self.policy.q_table[current_state][action]
        target = reward + self.gamma * max(self.policy.q_table[next_state])
        q_new = q_old + self.learning_rate * (target - q_old)

        # Update the Q-table
        self.policy.q_table[current_state][action] = q_new

    def learn(
        self: SelfQ, total_timesteps: int, callback: MaybeCallback = None, log_interval: int = 1,
        tb_log_name: str = "run", reset_num_timesteps: bool = True, progress_bar: bool = False,
        log_every_timestep: Optional[bool] = False
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
            rollout = self.collect_rollouts(callback, log_interval, log_every_timestep)
        
            if rollout.continue_training is False:
                break

            # Train on the collected data
            self.train()

        callback.on_training_end()

        return self

    def _setup_learn(
        self: SelfQ, total_timesteps: int, callback: MaybeCallback = None,
        reset_num_timesteps: bool = True, tb_log_name: str = "run",
        progress_bar: bool = False
    ) -> Tuple[int, BaseCallback]:
        # Resets the environment, sets-up callbacks and prepares tensorboard logging
        total_timesteps, callback = super()._setup_learn(
            total_timesteps, callback, reset_num_timesteps, tb_log_name, progress_bar
        )

        # We always work with one environment, so we can extract the first observation
        self._last_obs = self._last_obs[0]

        # Save reference to tensorboard formatter object
        self._tb_formatter = next(
            fmt for fmt in self.logger.output_formats if isinstance(fmt, TensorBoardOutputFormat)
        )

        return total_timesteps, callback

    def _dump_logs(self: SelfQ) -> None:
        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)

        self.logger.record("time/episodes", self._episode_num, exclude="tensorboard")

        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        
        self.logger.record("time/fps", fps, exclude="tensorboard")
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")

        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))

        # Pass the number of timesteps for tensorboard
        self.logger.dump(step=self.num_timesteps)

    def _included_save_params(self: SelfQ) -> List[str]:
        return [
            'learning_rate', 'gamma', 'seed', 'exploration_rate', 'exploration_initial_eps',
            'exploration_final_eps', 'exploration_fraction', 'use_sb3_env'
        ]

    def save(self: SelfQ, path: Union[str, pathlib.Path, io.BufferedIOBase], included: Optional[Iterable[str]] = []) -> None:
        # Generate an initial unique model id
        model_id = 1
        
        # Collect the number of previous times models with this configuration have been saved
        model_paths = list(dir for dir in glob.glob(f'{path}_*') if os.path.isdir(dir))

        # Check if there are actually any previous models
        if len(model_paths) != 0:
            # Extract the id of the last model we created to make a new unique id
            model_id += sorted(map(lambda name: int(name.rsplit('_', 1)[1]), model_paths))[-1]

        # Create a unique path name
        path = f'{path}_{model_id}'

        # Prepare the locations to save the parameters and Q-table to
        q_table_path = os.path.join(path, 'q_table.npy')
        q_param_path = os.path.join(path, 'q_params.npy')

        # Copy parameter list so we don't mutate the original dict
        data = self.__dict__.copy()

        # Exclude is union of specified parameters (if any) and standard exclusions
        included = set(included).union(self._included_save_params())
        keys = list(data.keys())

        # Remove the parameters entries that are not to be included
        for param_name in keys:
            if param_name not in included:
                data.pop(param_name, None)

        # Savely create the required directories for writing
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

        # Save the Q-table and model parameters
        np.save(q_table_path, self.policy.q_table)
        np.save(q_param_path, data)

    @classmethod
    def load(cls: Type[SelfQ], path: Union[str, pathlib.Path, io.BufferedIOBase], env: GymEnv) -> None:
        # Prepare the locations to load the parameters and Q-table from
        q_table_path = os.path.join(path, 'q_table.npy')
        q_param_path = os.path.join(path, 'q_params.npy')

        # Load the Q-table and model parameters
        q_table = np.load(q_table_path, allow_pickle=True)
        data = np.load(q_param_path, allow_pickle=True).item()

        # Instantiate a new model
        model = cls(env=env, learning_rate=None, init_setup_model=False)

        # Update the model parameters
        for key, value in data.items():
            setattr(model, key, value)
        
        model._setup_model()

        # Overwrite the initialized Q-table with the saved one.
        model.policy.q_table = q_table

        return model

    def _tb_write(self, tag: str, value: Union[str, float], timestep: int) -> None:
        # Writing to formatter depends on `value` type
        if isinstance(value, str):
            self._tb_formatter.writer.add_text(tag, value, timestep)
        else:
            self._tb_formatter.writer.add_scalar(tag, value, timestep)

        # Write the value to an output file
        self._tb_formatter.writer.flush()
