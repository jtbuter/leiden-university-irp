import matplotlib.pyplot as plt
import numpy as np

import irp
import irp.utils
import irp.envs as envs

from irp.policies.mask_tiled_policy import MaskTiledQ
from irp.envs.sahba_env import Env
from irp.envs.base_env import UltraSoundEnv
from irp.wrappers.masking import ActionMasker
from irp.wrappers.tiled import Tiled
from irp.agents.qlearning import Qlearning
from irp.agents.sarsa import Sarsa

def eval(locals_):
    instance = locals_['self']

    print(irp.utils.evaluate(instance.environment, instance.policy))

callback = {
    'interval': 10,
    'callback': eval
}

image_parameters = {
    'subimage_width': 32,
    'subimage_height': 16,
    'overlap': 0.5
}
neighborhood_parameters = {
    'n_size': 1,
    'overlap': image_parameters['overlap'],
    'neighborhood': 'neumann'
}
tiling_parameters = {
    'tiles_per_dim': (4, 4, 2),
    'tilings': 64,
    'limits': [(0, 1), (0, 1), (0, 4)]
}
agent_parameters = {
    'alpha': 0.8,
    'max_t': np.inf,
    'max_e': 2000,
    'eps_max': 0.6,
    'eps_min': 0.6,
    'eps_frac': 0.001,
    'gamma': 0.95,
    'callback': callback
}
environment_parameters = {
    'n_thresholds': 6,
    'openings': [0]
}

(image, truth), (t_image, t_truth) = irp.utils.stacked_read_sample('case10_10.png', 'case10_11.png', median_size=7)
subimages, sublabels, t_subimages, t_sublabels = irp.utils.extract_subimages(
    image, truth, t_image, t_truth, **image_parameters
)
coord = (256, 224)

sample_id = irp.utils.coord_to_id(coord, image.shape, **image_parameters)
sample, label = subimages[sample_id], sublabels[sample_id]
t_sample, t_label = t_subimages[sample_id], t_sublabels[sample_id]

environment = Env(sample, label, **environment_parameters)
environment = Tiled(environment, **tiling_parameters)
environment = ActionMasker(environment)

t_environment = Env(t_sample, t_label, **environment_parameters)
t_environment = Tiled(t_environment, **tiling_parameters)
t_environment = ActionMasker(t_environment)

agent = Qlearning(environment)
policy = agent.learn(**agent_parameters)

state, info = t_environment.reset(ti=0, vi=0)

for i in range(10):
    print(policy.values(state)[t_environment.action_mask()])

    action = policy.predict(state, t_environment.action_mask, deterministic=True)
    state, reward, done, info = t_environment.step(action)

    print(done, info)

    irp.utils.show(t_environment.bitmask)