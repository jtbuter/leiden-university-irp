from gym.wrappers import TimeLimit
import numpy as np
import os

import irp
from irp.manager.manager import ExperimentManager
from irp import utils
from irp.wrappers import Discretize
from irp.envs import Paper2008UltraSoundEnv
from irp.callbacks import StopOnDone
from irp.q import Q
from irp import utils

def parse_highs(area, compactness, objects, label):
    height, width = label.shape

    if objects == "normalize":
        objects = int(np.ceil(width / 2) * np.ceil(height / 2))

    return {
        'area': area,
        'compactness': compactness,
        'objects': objects
    }

def train(
    manager: ExperimentManager, train_image_path: str, test_image_path: str,
    learning_rate: float, gamma: float, exploration_fraction: float,
    exploration_rate: float, num_thresholds: int, vjs: tuple, bins: tuple,
    episode_length: int, lows: dict, highs: dict, num_timesteps: int,
    stop_on_done: bool
):
    data = utils.make_sample_label(train_image_path, test_image_path)

    train_image, train_label = data[0]
    test_image, test_label = data[1]

    # Parse the bin division descriptors
    highs = parse_highs(**highs, label=train_label)

    # Update the experimental settings
    manager.set_value('highs', highs)

    # Construct the path where experiments are saved
    experiments_folder = os.path.join(
        irp.ROOT_DIR, manager.experiment_root, manager.experiment_name
    )
    model_folder = os.path.join(experiments_folder, "model")

    # Store the log-files for tensorboard directly in the experiments folder,
    # instead of making a separate sub-directory
    tensorboard_log = os.path.join(experiments_folder)

    # Construct a tensorboard friendly experiment name
    tb_log_name = utils.params_to_modelname(
        **{key: manager.experiment[key] for key in [
            'learning_rate', 'gamma', 'exploration_rate', 'episode_length'
        ]}
    )

    # Initialize potential callback
    callback = None

    # Specifically evaluate for when `stop_on_done` is True
    if stop_on_done is True:
        # Create callback for stopping when the experiment is done
        callback = StopOnDone()

    # Initialize the environment
    env = Paper2008UltraSoundEnv(
        train_image, train_label, num_thresholds=num_thresholds, vjs=vjs
    )
    env = Discretize(env, lows, highs, bins)
    env = TimeLimit(env, episode_length)

    model = Q(
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_rate,
        tensorboard_log=tensorboard_log
    )

    # Start the learning process
    model.learn(num_timesteps, callback=callback, tb_log_name=tb_log_name)

    # Save the model
    model.save(model_folder)