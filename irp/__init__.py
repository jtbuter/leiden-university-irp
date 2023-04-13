import os
import pathlib

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
GIT_DIR = pathlib.Path(ROOT_DIR).parent.absolute()
