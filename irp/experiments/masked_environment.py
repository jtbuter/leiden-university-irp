import matplotlib.pyplot as plt
import numpy as np

import irp
import irp.utils
import irp.envs as envs

from irp.envs.env import Env
from irp.envs.base_env import UltraSoundEnv
from irp.wrappers.masking import ActionMasker
from irp.wrappers.tiled import Tiled
from irp.agents.qlearning import Qlearning
from irp.agents.sarsa import Sarsa

results = []

def eval(local):
    pass
    # d_sim, done, bitmask = irp.utils.evaluate(environment, local['self'].policy, max_steps=environment.n_thresholds)

    # results.append(d_sim)

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
coord = (320, 272)

sample_id = irp.utils.coord_to_id(coord, image.shape, **image_parameters)
sample, label = subimages[sample_id], sublabels[sample_id]
t_sample, t_label = t_subimages[sample_id], t_sublabels[sample_id]

environment = Env(sample, label, **environment_parameters)
environment = Tiled(environment, **tiling_parameters)
environment = ActionMasker(environment)

agent = Sarsa(environment)
pi = agent.learn(**agent_parameters, callback=callback)

ti = (0, environment.n_thresholds - 1)
ti = (1, 5)

done = False
state, info = environment.reset(ti=ti)

print(pi.values(state)[np.where(environment.guidance_mask() == True)[0]])

print(np.asarray(environment.action_mapping)[np.where(environment.guidance_mask() == True)[0]])

print(info['d_sim'], environment.d_sim_opt)
print('Internal ti on initialization', environment.ti_left, environment.ti_right)
irp.utils.show(environment.bitmask, environment.label)

states = []
values = []

for step in range(30):
    action = pi.predict(state, environment.action_mask)
    state, reward, done, info = environment.step(action)

    print('Action', action, environment.action_mapping[action])
    print('Internal ti after action', environment.ti_left, environment.ti_right)
    print('Step status', reward, 'Target' if done else 'Searching', info)

    qvalues = pi.values(state)[environment.action_mask()]

    print('Min, max Q-value', min(qvalues), max(qvalues))

    states.append(state)
    values.append(max(qvalues))

    states_ = irp.utils.simplify_sequence(states, states)
    path = irp.utils.find_repeating_path(list(map(tuple, states_)))

    if len(path):
        best_vals = [values[p] for p in path]

        print('Assumed target', min(best_vals))

        break

# print(irp.utils.simplify_sequence(states, values))

# t_environment = Env(t_sample, t_label, **environment_parameters)
# t_environment = Tiled(t_environment, **tiling_parameters)
# t_environment = ActionMasker(t_environment)

raise Exception
values = [pi.values(t_environment.reset(ti=i)[0]) for i in range(t_environment.n_thresholds)]
k = 0

for i, value in enumerate(values):
    t_environment.reset(ti=i)

    rep = ''

    if i == j:
        rep += 'x'

    if envs.utils.compute_dissimilarity(t_label, t_environment.bitmask) == t_environment.d_sim_opt:
        rep += 'y'

    print(rep, UltraSoundEnv.observation(t_environment.bitmask), value)
        
# print("\n".join(list(map(str, values))))

print()

state, info = t_environment.reset(ti=0)
print(UltraSoundEnv.observation(t_environment.bitmask))

irp.utils.show(t_environment.bitmask)

for i in range(20):
    action = pi.predict(state, lambda: t_environment.action_mask())

    state, reward, done, info = t_environment.step(action)
    print(UltraSoundEnv.observation(t_environment.bitmask))

    irp.utils.show(t_environment.bitmask)
