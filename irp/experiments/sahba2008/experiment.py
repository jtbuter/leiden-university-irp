import numpy as np
import os

from stable_baselines3.common.callbacks import CallbackList

from irp.manager.manager import ExperimentManager
from irp.callbacks import (
    StopOnDoneCallback, LogNStepsCallback, EvalCallback
)
from irp.q import Q
import irp.utils
import irp

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
            'learning_rate', 'gamma', 'exploration_final_eps', 'episode_length'
        ]}
    )

    # Initialize the train and evaluation environments
    env = irp.utils.setup_environment(train_image, train_label, num_thresholds, vjs, lows, highs, bins, episode_length)
    eval_env = irp.utils.setup_environment(test_image, test_label, num_thresholds, vjs, lows, highs, bins, episode_length)

    # Initialize potential callback
    callback_list = []

    # Specifically evaluate for when `stop_on_done` is True
    if stop_on_done is True:
        # Create callback for stopping when the experiment is done
        callback_list.append(StopOnDoneCallback())

    # Log the reward every n steps
    callback_list.append(LogNStepsCallback(freq=100))

    # Add the evaluation callback to the list of callbacks to execute
    callback_list.append(EvalCallback(eval_env, eval_freq=100, n_eval_episodes=100))

    # Set-up callback that executes all of the callbacks
    callback = CallbackList(callbacks=callback_list)

    # Set-up the Q-learning model
    model = Q(
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps, # Acts as an exploration rate
        tensorboard_log=tensorboard_log
    )

    # Start the learning process
    model.learn(num_timesteps, callback=callback, tb_log_name=tb_log_name)

    # Save the model
    model.save(model_folder)

    return model

if __name__ == "__main__":
    # Initialize the manager
    manager = ExperimentManager(
        experiment_root='results',
        experiment_name='sahba2008',
        code_file=__file__,
        config_file=os.path.join(irp.ROOT_DIR, 'experiments/sahba2008/conf.yaml'),
        tb_friendly=True
    )

    manager.set_value('stop_on_done', False)
    manager.set_value('episode_length', 1)
    manager.set_value('num_timesteps', 10000)

    model = manager.start(train)

    # experiment = manager.experiment
    # test_image, test_label = experiment['data'][1]
    # num_thresholds = experiment['num_thresholds']
    # vjs = experiment['vjs']
    # lows = experiment['lows']
    # highs = experiment['highs']
    # bins = experiment['bins']
    # episode_length = experiment['episode_length']

    # # Initialize the environment
    # eval_env = Sahba2008UltraSoundEnv(test_image, test_label, num_thresholds, vjs)

    # # Cast continuous values to bins
    # eval_env = Discretize(eval_env, lows, highs, bins)
    
    # # Set a maximum episode length
    # eval_env = TimeLimit(eval_env, episode_length)

    # mean_reward, std_reward = irp.utils.evaluate_policy(
    #     model, eval_env, n_eval_episodes=20
    # )

    # model.logger.record("eval/mean_reward", float(mean_reward))
    # model.logger.dump(model.num_timesteps)
