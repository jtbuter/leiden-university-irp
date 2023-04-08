import os
import argparse
import yaml
import re
import json

import irp
from irp import utils

class ExperimentManager:
    parser = argparse.ArgumentParser()
    allowed_arg_types = ['str', 'float', 'int', 'bool']

    def __init__(self, experiment_name = None, experiment_root = None) -> None:
        if experiment_root is None:
            experiment_root = self._get_experiment_root()

        if experiment_name is None:
            experiment_name = self._get_experiment_name()

        self.experiment_root = experiment_root
        self.experiment_name = experiment_name

        # Read config file and parse cli arguments
        self._setup_experiment()

    def _get_experiment_name(self):
        self.parser.add_argument(
            '--experiment-name', default="", type=str,
            help="Defines the basename of the experiment"
        )

        args, _ = self.parser.parse_known_args()

        return args.experiment_name

    def _get_experiment_root(self):
        self.parser.add_argument(
            '--experiment-root', default="experiments", type=str,
            help="Defines the root all experiments"
        )

        args, _ = self.parser.parse_known_args()

        return args.experiment_root

    def _setup_experiment(self) -> None:
        # Create the path to a possible .YAML configuration file
        cfg_path = os.path.join(
            irp.ROOT_DIR, self.experiment_root, self.experiment_name, 'conf.yaml'
        )

        self.experiment = {}

        # See if a configuration file exists
        if os.path.isfile(cfg_path):
            with open(cfg_path, 'r') as cfg:
                config = yaml.safe_load(cfg)

                assert 'experiment' in config, "No experiment definition exists in conf.yaml"

                self.experiment.update(config['experiment'])

        _, unknown = self.parser.parse_known_args()

        for arg in unknown:
            # Defines a value we want to override the config with
            if not arg.startswith("--experiment."): continue

            # Extract the raw name and value
            raw_name, value = arg.rsplit('=', 1)

            # Remove the unnecessary suffix
            raw_name = re.sub('^--experiment\.', '', raw_name)

            # Variable type must be indicated
            name, builtin = raw_name.rsplit(':', 1)

            assert builtin in self.allowed_arg_types, f"Unrecognized argument type: {builtin}"

            # Save the value to experiments
            self.experiment[name] = utils.str_to_builtin(value, builtin)

    def __str__(self) -> str:
        definition = {
            'Root directory:': self.experiment_root,
            'Experiment name': self.experiment_name,
            'Experimental tree': self.experiment
        }

        return json.dumps(definition, indent = 4)

manager = ExperimentManager()

print(manager)