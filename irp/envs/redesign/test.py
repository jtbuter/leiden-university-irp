from irp.envs.redesign.trus_env import TrusEnv
from irp.wrappers import Discretize
from irp import utils

import numpy as np

image_path = 'case10_10.png'
x, y = utils.make_sample_label(image_path)[0]

trus_env = TrusEnv(x, y, 1, 1)
disc_env = Discretize(trus_env, [0, 0, 0], [1, 1, 1], (10, 10, 10))

bit_mask = np.zeros((32, 16), dtype=np.uint8)
bit_mask[0:10, 0:10] = 1
obs = trus_env.observation(bit_mask)

print("square", obs)

bit_mask = np.zeros((32, 16), dtype=np.uint8)
obs = trus_env.observation(bit_mask)

print("zeros", obs)

bit_mask = np.ones((32, 16), dtype=np.uint8)
obs = trus_env.observation(bit_mask)

print("ones", obs)

bit_mask = np.zeros((32, 16), dtype=np.uint8)
bit_mask[0, 0] = 1
obs = trus_env.observation(bit_mask)

print("pixel", obs)
