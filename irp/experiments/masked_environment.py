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

results = []

def eval(local):
    d_sim, done, bitmask = irp.utils.evaluate(environment, local['self'].policy, max_steps=environment.n_thresholds)

    results.append(d_sim)

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
    'tiles_per_dim': (2, 2, 1),
    'tilings': 16,
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
    'n_thresholds': 8,
    'opening': 8
}

(image, truth), (t_image, t_truth) = irp.utils.read_sample('case10_10.png'), irp.utils.read_sample('case10_11.png')
subimages, sublabels, t_subimages, t_sublabels = irp.utils.extract_subimages(
    image, truth, t_image, t_truth, **image_parameters
)
coord = (288 - 16, 272 + 8)

sample_id = irp.utils.coord_to_id(coord, image.shape, **image_parameters)
sample, label = subimages[sample_id], sublabels[sample_id]
t_sample, t_label = t_subimages[sample_id], t_sublabels[sample_id]

print(np.min(sample), np.max(sample))

print(envs.utils.get_intensity_spectrum(sample, environment_parameters['n_thresholds'], add_minus=True))

environment = Env(sample, label, **environment_parameters)
environment = Tiled(environment, **tiling_parameters)
environment = ActionMasker(environment)

t_environment = Env(t_sample, t_label, **environment_parameters)
t_environment = Tiled(t_environment, **tiling_parameters)
t_environment = ActionMasker(t_environment)

intensities = envs.utils.get_intensity_spectrum(sample, environment_parameters['n_thresholds'], add_minus=True)

bitmasks = []
j = 0

for f, intensity in enumerate(intensities):
    bitmask = irp.utils.apply_action_sequence(sample, (intensity, 8), (envs.utils.apply_threshold, envs.utils.apply_opening))

    if envs.utils.compute_dissimilarity(label, bitmask) == environment.d_sim_opt:
        j = f
        print('x', UltraSoundEnv.observation(bitmask))
        irp.utils.show(bitmask)
    else:
        print(UltraSoundEnv.observation(bitmask))

    bitmasks.append(bitmask)

print()

irp.utils.show(*bitmasks)

irp.utils.show(label)

named_actions = ['decrease', 'increase']
agent = Qlearning(environment)
pi = agent.learn(**agent_parameters, callback=callback)

for i in range(environment.n_thresholds):
    state, info = environment.reset(ti=i)
    bitmask = environment.bitmask

    print(
        'State:', UltraSoundEnv.observation(bitmask), '\n',
        'Q(s, a):', pi.values(state), '\n',
        'Best action:', named_actions[pi.predict(state, lambda: environment.action_mask())], '\n',
        'Action mask:', environment.action_mask(), '\n',
        'Info:', info['d_sim']
    )
    print(state)


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
