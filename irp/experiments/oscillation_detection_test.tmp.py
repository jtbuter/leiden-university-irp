from typing import Generic, TypeVar, Union
import irp.utils

from irp.policies.mask_tiled_policy import MaskTiledQ
from irp.wrappers.tiled import Tiled
from irp.wrappers.masking import ActionMasker
from irp.wrappers.binned import Binned
from irp.envs.sahba_env import Env
from irp.envs.base_env import UltraSoundEnv
from irp.agents.qlearning import Qlearning
from irp.agents.sarsa import Sarsa
from irp.policies.policy import Q

import numpy as np

image_parameters = {
    'subimage_width': 32,
    'subimage_height': 16,
    'overlap': 0.5
}
tiling_parameters = {
    'tiles_per_dim': (4, 4, 2),
    'tilings': 64,
    'limits': [(0, 1), (0, 1), (0, 4)]
}
agent_parameters = {
    'alpha': 0.2,
    'max_t': np.inf,
    'max_e': 2000,
    'eps_max': 0.6,
    'eps_min': 0.6,
    'eps_frac': 0.001,
    'gamma': 0.95,
}
environment_parameters = {
    'n_thresholds': 6,
    'openings': [0, 2, 5]
}

(image, truth), (t_image, t_truth) = irp.utils.stacked_read_sample('case10_10.png', 'case10_11.png', median_size=7)
subimages, sublabels, t_subimages, t_sublabels = irp.utils.extract_subimages(
    image, truth, t_image, t_truth, **image_parameters
)

coord = (256, 176)
sample_id = irp.utils.coord_to_id(coord, image.shape, **image_parameters)

sample, label = subimages[sample_id], sublabels[sample_id]
t_sample, t_label = t_subimages[sample_id], t_sublabels[sample_id]

environment = Env(sample, label, **environment_parameters)
environment = Tiled(environment, **tiling_parameters)
environment = ActionMasker(environment)

gamma = 0.99
policy = MaskTiledQ(environment.n_features, environment.n_tiles, environment.action_space.n, alpha=1)

bitmasks = []
states = []
values = []
done_state = None

# First iteration
state, info = environment.reset(ti=0, vi=0)

action = environment.action_mapping.index((1, 1)); next_state, reward, done, info = environment.step(action);
target = reward + gamma * max(policy.values(next_state)); policy.update(state, action, target); state = next_state

action = environment.action_mapping.index((1, 1)); next_state, reward, done, info = environment.step(action);
target = reward + gamma * max(policy.values(next_state)); policy.update(state, action, target); state = next_state

action = environment.action_mapping.index((1, 0)); next_state, reward, done, info = environment.step(action);
target = reward + gamma * max(policy.values(next_state)); policy.update(state, action, target); state = next_state

# Second iteration
state, info = environment.reset(ti=0, vi=0)

action = environment.action_mapping.index((1, 1)); next_state, reward, done, info = environment.step(action);
target = reward + gamma * max(policy.values(next_state)); policy.update(state, action, target); state = next_state

action = environment.action_mapping.index((1, 1)); next_state, reward, done, info = environment.step(action);
target = reward + gamma * max(policy.values(next_state)); policy.update(state, action, target); state = next_state

action = environment.action_mapping.index((1, 0)); next_state, reward, done, info = environment.step(action);
target = reward + gamma * max(policy.values(next_state)); policy.update(state, action, target); state = next_state

# Start backtracking
state, info = environment.reset(ti=0, vi=0)

states.append(tuple(state)); values.append(np.min(policy.values(state))); bitmasks.append(environment.bitmask.copy())

action = environment.action_mapping.index((1, 1)); next_state, reward, done, info = environment.step(action);
state = next_state

states.append(tuple(state)); values.append(np.min(policy.values(state))); bitmasks.append(environment.bitmask.copy())

action = environment.action_mapping.index((1, 1)); next_state, reward, done, info = environment.step(action);
state = next_state

states.append(tuple(state)); values.append(np.min(policy.values(state))); bitmasks.append(environment.bitmask.copy())

# Here we are in the terminal state
action = environment.action_mapping.index((1, 0)); next_state, reward, done, info = environment.step(action);
state = next_state

states.append(tuple(state)); values.append(np.min(policy.values(state))); bitmasks.append(environment.bitmask.copy())

action = environment.action_mapping.index((-1, -1)); next_state, reward, done, info = environment.step(action);
state = next_state

states.append(tuple(state)); values.append(np.min(policy.values(state))); bitmasks.append(environment.bitmask.copy())

action = environment.action_mapping.index((1, 0)); next_state, reward, done, info = environment.step(action);
state = next_state

states.append(tuple(state)); values.append(np.min(policy.values(state))); bitmasks.append(environment.bitmask.copy())

action = environment.action_mapping.index((0, 1)); next_state, reward, done, info = environment.step(action);
state = next_state

states.append(tuple(state)); values.append(np.min(policy.values(state))); bitmasks.append(environment.bitmask.copy())

if done:
    print('x')
    
    done_state = state

filtered_states = irp.utils.simplify_sequence(states, states)
cycle = irp.utils.find_repeating_path(filtered_states)

print(values)

states, values = np.asarray(states), np.asarray(values)

# Check if we've found a cycle
if len(cycle):
    predicted_state = states[cycle[0] + np.argmin(values[cycle])]

    print(done_state)
    print(predicted_state)

# print(values)