from matplotlib import pyplot as plt
import numpy as np

import irp
import irp.utils
import irp.envs as envs
import irp.wrappers as wrappers
from irp.envs.ultrasound.ultra_sound_env import UltraSoundEnv

coords = [(256, 176), (288, 176), (240, 184), (256, 184), (224, 192), (240, 192), (304, 192), (224, 200), (320, 200), (224, 208), (320, 208), (208, 216), (208, 224), (336, 224), (208, 232), (192, 240), (336, 240), (192, 248), (304, 248), (208, 264), (256, 264), (272, 264), (208, 272), (224, 272), (240, 272), (256, 272), (288, 272), (240, 280), (288, 280), (304, 280), (304, 288), (320, 288)]
