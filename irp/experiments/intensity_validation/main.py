from irp.callbacks import LogNStepsCallback
from gym.wrappers import TimeLimit
import numpy as np
import os
from stable_baselines3.common.callbacks import CallbackList

import irp
import irp.utils
from irp.manager.manager import ExperimentManager
from irp.wrappers import Discretize
from irp.envs import Paper2008UltraSoundEnv
from irp.callbacks import StopOnDoneCallback
from irp.q import Q

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
    exploration_final_eps: float, num_thresholds: int, vjs: tuple, bins: tuple,
    episode_length: int, lows: dict, highs: dict, num_timesteps: int,
    stop_on_done: bool
):
    data = irp.utils.make_sample_label(train_image_path, test_image_path)

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
    tb_log_name = irp.utils.params_to_modelname(
        **{key: manager.experiment[key] for key in [
            'learning_rate', 'gamma', 'exploration_final_eps', 'episode_length', 'num_timesteps'
        ]}
    )

    # Initialize potential callback
    callback = CallbackList(callbacks=[
        StopOnDoneCallback(), LogNStepsCallback(freq=100)
    ])

    # Initialize the environment
    env = Paper2008UltraSoundEnv(train_image, train_label, num_thresholds, vjs)
    env = Discretize(env, lows, highs, bins)
    env = TimeLimit(env, episode_length)

    # Initialize the model
    model = Q(
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
        tensorboard_log=tensorboard_log
    )

    # Start the learning process
    model.learn(num_timesteps, callback=callback, tb_log_name=tb_log_name)

# Get the name of the current file
code_file = __file__

# Explicitly use path to current file, instead of relative
cfg_file = os.path.join(irp.ROOT_DIR, 'experiments/sahba2008/conf.yaml')

# Initialize the manager
manager = ExperimentManager(
    experiment_root='results',
    experiment_name='intensity_validation',
    code_file=code_file, config_file=cfg_file,
    tb_friendly=True, verbose=0
)

# Sets the episode length equal to the number of time-steps to 
manager.set_value('episode_length', 50)
manager.set_value('num_timesteps', 5000)
manager.set_value('exploration_final_eps', 1)
# manager.set_value('exploration_fraction', 0)
# manager.set_value('exploration_fraction', 0.5)

manager.start(train)
