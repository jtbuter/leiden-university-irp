import glob
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
import pathlib
from pygments import highlight, lexers, formatters

import irp
from irp import utils

class ExperimentManager:
    parser = argparse.ArgumentParser()
    allowed_arg_types = ['str', 'float', 'int', 'bool']

    def __init__(
        self, code_file: str, experiment_name: str = None,
        experiment_root: str = None, experiment: dict = None,
        config_file: str = None, tb_friendly: bool = False,
        verbose: int = 0
    ) -> None:
        """
        code_file: Path of the file executing the code
        experiment_name: Name of the folder to save the results to
        experiment_root: Location containing the `experiment_name` folder
        experiment: Dict containing user-defined experimental set-up
        config_file: Path to a configuration file
        tb_friendly: Should we generate a unique name for the meta.npy file
        verbose: Level of warnings we should display
        """

        if experiment_root is None:
            experiment_root = self._get_experiment_root()

        if experiment_name is None:
            experiment_name = self._get_experiment_name(code_file)

        self.experiment_root = experiment_root
        self.experiment_name = experiment_name
        self.experiment = experiment
        self.tb_friendly = tb_friendly
        self.verbose = verbose

        # Ensures manager is being used properly, where code_file is the name
        # of the file executing the experiment
        self._build(code_file)

        # Read config file and parse cli arguments
        self._setup_experiment(config_file)

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
        if code_file in uncommited_files and self.verbose > 0:
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

    def _get_experiment_name(self, code_file) -> str:
        self.parser.add_argument(
            '--experiment-name', default=None, type=str,
            help="Defines the basename of the experiment"
        )

        # Don't raise an error when unknown arguments are passed
        args, _ = self.parser.parse_known_args()

        experiment_name = args.experiment_name

        # Use parent directory of file using the manager to store experiments
        if experiment_name is None or experiment_name == "":
            code_file = pathlib.Path(os.path.abspath(code_file))
            code_folder = os.path.basename(code_file.parent)

            experiment_name = code_folder

        return experiment_name

    def _get_experiment_root(self) -> str:
        self.parser.add_argument(
            '--experiment-root', default="results", type=str,
            help="Defines the root all experiments"
        )

        # Don't raise an error when unknown arguments are passed
        args, _ = self.parser.parse_known_args()

        return args.experiment_root

    def _setup_experiment(self, cfg_path) -> None:
        # Test if during construction an experiment was passed by the user
        if self.experiment is None:
            self.experiment = {}

        if cfg_path is None:
            # Create the path to a possible .YAML configuration file
            cfg_path = os.path.join(
                irp.ROOT_DIR, self.experiment_root, self.experiment_name, 'conf.yaml'
            )

        # See if a configuration file exists
        if os.path.isfile(cfg_path):
            with open(cfg_path, 'r') as cfg:
                config = yaml.safe_load(cfg)

                assert 'experiment' in config, "No experiment definition exists in conf.yaml"

                self.experiment.update(config['experiment'])
                self.metadata['details']['config_file'] = cfg_path

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
        # defined by the user
        self.metadata['user_properties'] = {}

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
        
        meta_folder = os.path.join(
            irp.ROOT_DIR, self.experiment_root, self.experiment_name
        )
        meta_name = 'meta'

        # We should create a unique filename for the meta-file; tensorboard style
        if self.tb_friendly:
            # Generate an initial unique model id
            model_id = 1

            # Collect the number of previous times metas with this configuration have been saved
            model_paths = glob.glob(os.path.join(meta_folder, 'meta_*.npy'))

            # Check if there are actually any previous metas
            if len(model_paths) != 0:
                # Extract the id of the last model we created to make a new unique id.
                model_id += sorted(map(
                    lambda name: int(re.search('meta_(\d+)\.npy', name).group(1)), model_paths
                ))[-1]

            # Append the unique identifier to the meta_name
            meta_name += f'_{model_id}'

        # Add the extension to the meta_name
        meta_name += '.npy'
        meta_path = os.path.join(meta_folder, meta_name)

        # Checking if meta-file in this location already exists (unnecessary
        # when using tb_friendly mode)
        if os.path.isfile(meta_path) and self.verbose > 0:
            warnings.warn('You\'re overwriting an existing meta-data file!')

        # Make sure all the required directories exist/are created
        pathlib.Path(meta_folder).mkdir(parents=True, exist_ok=True)

        # Save the data
        np.save(meta_path, _metadata, allow_pickle=True)

    def set_property(self, name, value):
        # Save a user-defined property to our metadata
        self.metadata['user_properties'][name] = value
       
    def __str__(self) -> str:
        definition = {
            'Root directory:': self.experiment_root,
            'Experiment name': self.experiment_name,
            'Experimental tree': self.experiment,
            'Meta-data': self.metadata
        }

        # Get a JSON representation of our manager
        json_repr = json.dumps(definition, indent=4)

        # Return a syntax-highlighted version
        return highlight(
            json_repr, lexers.JsonLexer(), formatters.TerminalFormatter()
        )
