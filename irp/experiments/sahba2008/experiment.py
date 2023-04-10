from pathlib import Path
import os
from irp.manager.manager import ExperimentManager

manager = ExperimentManager(__file__)

def train(manager: ExperimentManager = None):
    manager.set_property('training_data', 'case10_11.png')
    manager.set_property('training_data', 'case10_11.png')


# manager.start(train)
