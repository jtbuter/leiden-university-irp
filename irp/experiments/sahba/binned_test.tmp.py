from typing import Generic, TypeVar, Union
import irp.utils

from irp.wrappers.masking import ActionMasker
from irp.wrappers.binned import Binned
from irp.envs.sahba_env import Env
from irp.envs.base_env import UltraSoundEnv
from irp.agents.qlearning import Qlearning
from irp.policies.policy import Q

import numpy as np

image_parameters = {
    'subimage_width': 32,
    'subimage_height': 16,
    'overlap': 0.5
}
binning_parameters = {
    'bins_per_dim': (20, 20, 20),
    'limits': [(0, 1), (0, 1), (0, 19)]
}
agent_parameters = {
    'alpha': 0.8,
    'max_t': np.inf,
    'max_e': 2000,
    'eps_max': 0.6,
    'eps_min': 0.6,
    'eps_frac': 0.001,
    'gamma': 0.6,
}
environment_parameters = {
    'n_thresholds': 6,
    'opening': 8
}

(image, truth), (t_image, t_truth) = irp.utils.read_sample('case10_10.png'), irp.utils.read_sample('case10_11.png')
subimages, sublabels, t_subimages, t_sublabels = irp.utils.extract_subimages(
    image, truth, t_image, t_truth, **image_parameters
)

coord = (192, 208)

sample_id = irp.utils.coord_to_id(coord, image.shape, **image_parameters)
sample, label = subimages[sample_id], sublabels[sample_id]
t_sample, t_label = t_subimages[sample_id], t_sublabels[sample_id]

environment = Env(sample, label, **environment_parameters)
environment = Binned(environment, **binning_parameters)
environment = ActionMasker(environment)

environment: Union[Env, Binned, ActionMasker]

best_actions = {}

for i in range(environment.n_thresholds):
    for vj in environment.morphological_actions:
        state, info = environment.reset(ti=i, vj=vj)
        d_sim = info['d_sim']

        for a in range(len(environment.action_mapping)):
            bitmask = environment.transition(a)

        print(i, vj, state, info)

        irp.utils.show(environment.bitmask)

# agent = Qlearning(environment)
# pi = agent.learn(policy_cls=Q, **agent_parameters)

# # t_environment = Env(t_sample, t_label, **environment_parameters)
# t_environment = Env(sample, label, **environment_parameters)
# t_environment = Binned(t_environment, **binning_parameters)
# t_environment = ActionMasker(t_environment)

# state, info = t_environment.reset(ti=0, vj=0)

# print(t_environment.action_mapping)

# irp.utils.show(t_environment.bitmask)
# print(state, pi.values(state))

# for i in range(10):
#     action = pi.predict(state, t_environment.action_mask)
#     print(t_environment.action_mapping[action])

#     state, reward, done, info = t_environment.step(action)
    
#     print(state, pi.values(state))

#     irp.utils.show(t_environment.bitmask)
