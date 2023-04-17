import os
from gym.wrappers import TimeLimit

from irp.envs import Sahba2008RangedEnv
from irp.manager import ExperimentManager
from irp.wrappers import Discretize
import irp.utils
import irp

# Load the experiment manager to get access to some environment parameters
manager = ExperimentManager(
    experiment_root='results',
    experiment_name='sahba2008',
    code_file=__file__,
    config_file=os.path.join(irp.ROOT_DIR, 'experiments/sahba2008/conf.yaml'),
    tb_friendly=True
)
experiment = manager.experiment
image, label = irp.utils.make_sample_label(experiment['train_image_path'])[0]

# Extract the parameters we need for setting up the environment
num_thresholds, vjs, lows, highs, bins, episode_length = tuple(
    experiment[key] for key in [
        'num_thresholds', 'vjs', 'lows', 'highs', 'bins', 'episode_length'
    ]
)

# Parse the bin division descriptors
highs = irp.utils.parse_highs(**highs, label=label)

# Initialize the environment
env = Sahba2008RangedEnv(image, label, num_thresholds, vjs)

# Cast continuous values to bins
env = Discretize(env, lows, highs, bins)

# Set a maximum episode length
env = TimeLimit(env, episode_length)


