import os
import argparse
import yaml
import re
import json
from copy import deepcopy
import warnings
import numpy as np
import inspect
import git

import irp
from irp import utils

class ExperimentManager:
    parser = argparse.ArgumentParser()
    allowed_arg_types = ['str', 'float', 'int', 'bool']

    def __init__(
        self, code_file: str, experiment_name: str = None,
        experiment_root: str = None, experiment = None
    ) -> None:
        if experiment_root is None:
            experiment_root = self._get_experiment_root()

        if experiment_name is None:
            experiment_name = self._get_experiment_name()

        self.experiment_root = experiment_root
        self.experiment_name = experiment_name
        self.experiment = experiment

        # Ensures manager is being used properly, where code_file is the name
        # of the file executing the experiment
        self._build(code_file)

        # Read config file and parse cli arguments
        self._setup_experiment()

    def _build(self, code_file):
        # TODO: Use https://stackoverflow.com/a/52307120/10069642 solution for getting the executing file
        # Get full name of the file executing the experiment, up until the git-repo
        code_file = os.path.abspath(code_file)
        code_file = os.path.relpath(code_file, irp.GIT_DIR)

        # Off-load version control to git
        repo = git.Repo(irp.GIT_DIR)

        # Get all unstaged changes
        unstaged_files = list(diff.b_path for diff in repo.index.diff(None))

        # Get all staged changes
        staged_files = list(diff.b_path for diff in repo.head.commit.diff(None))

        # Get all untracked files
        untracked_files = list(repo.untracked_files)

        # Make sure the file calling the experiment has been committed
        uncommited_files = unstaged_files + staged_files + untracked_files

        # If the experiment hasn't been committed, warn the user
        if code_file in uncommited_files:
            warnings.warn('Running an uncommitted experiment file')

        # Get the id of the current commit
        commit_id = repo.head.object.hexsha
        branch_name = repo.active_branch.name

        # Saves the commit and branch of the experiment so we can always
        # recover the code
        self.metadata = {
            'details': {
                'commit_id': commit_id,
                'branch_name': branch_name
            }
        }

    def _get_experiment_name(self) -> str:
        self.parser.add_argument(
            '--experiment-name', default="experiments", type=str,
            help="Defines the basename of the experiment"
        )

        # Don't raise an error when unknown arguments are passed
        args, _ = self.parser.parse_known_args()

        return args.experiment_name

    def _get_experiment_root(self) -> str:
        self.parser.add_argument(
            '--experiment-root', default="experiments", type=str,
            help="Defines the root all experiments"
        )

        # Don't raise an error when unknown arguments are passed
        args, _ = self.parser.parse_known_args()

        return args.experiment_root

    def _setup_experiment(self) -> None:
        # Create the path to a possible .YAML configuration file
        cfg_path = os.path.join(
            irp.ROOT_DIR, self.experiment_root, self.experiment_name, 'conf.yaml'
        )

        if self.experiment is None:
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
            name = re.sub('^--experiment\.', '', raw_name)

            # Extract indicated variable type
            # TODO: Optionally remove this, we can also use eval
            # name, builtin = raw_name.rsplit(':', 1)

            # assert builtin in self.allowed_arg_types, f"Unrecognized argument type: {builtin}"

            # Save the value to experiments
            self.experiment[name] = utils.str_to_builtin(value)

        # Initialize a dictionary for storing optional additional meta-data
        self.metadata['properties'] = {}

    def start(self, user_function):
        # Avoid modifying the original experiment parameters
        _experiment = deepcopy(self.experiment)

        assert 'manager' not in _experiment, "No key named 'manager' is allowed"

        # Give function access to the current instance
        _experiment['manager'] = self

        # Unpack the experiment dictionary into named function arguments, and
        # run the experiment
        user_function(**_experiment)

        # Store any meta-data a user has defined
        if self.metadata:
            self._dump()

        return self

    def _dump(self):
        # Initialize storage for the configuration used by the experiment
        # and the optional user-defined meta-data
        _metadata = {
            'experiment': self.experiment,
            'metadata': self.metadata
        }
        
        meta_path = os.path.join(
            irp.ROOT_DIR, self.experiment_root, self.experiment_name, 'meta.npy'
        )

        if os.path.isfile(meta_path):
            warnings.warn('You\'re overwriting an existing meta-data file!')

        np.save(meta_path, _metadata, allow_pickle=True)

    def set_property(self, name, value):
        # Save a user-defined property to our metadata
        self.metadata['properties'][name] = value
       
    def __str__(self) -> str:
        definition = {
            'Root directory:': self.experiment_root,
            'Experiment name': self.experiment_name,
            'Experimental tree': self.experiment
        }

        return json.dumps(definition, indent = 4)