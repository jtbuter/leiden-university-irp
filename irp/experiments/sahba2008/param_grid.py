from sklearn.model_selection import ParameterGrid
import os

from irp.manager.manager import ExperimentManager
from irp.experiments.sahba2008.experiment import train
import irp

grid = ParameterGrid({
    'gamma': [0.6],
    'exploration_rate': [0.8, 0.9, 1.0]
})

# Get the name of this file
code_file = __file__

# Explicitly use path to current file, instead of relative
cfg_file = os.path.join(irp.ROOT_DIR, 'experiments/sahba2008/conf.yaml')

# Initialize the manager
manager = ExperimentManager(
    experiment_root='results',
    experiment_name='sahba2008',
    code_file=code_file, config_file=cfg_file,
    tb_friendly=True
)

# Iterate over all parameters
for param in grid:
    # Update the experiment to match the parameter grid
    for parameter, value in param.items():
        manager.set_value(parameter, value)

    manager.start(train)
