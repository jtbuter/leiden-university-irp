import numpy as np
import matplotlib.pyplot as plt

import irp.utils
import irp.wrappers as wrappers
import irp.envs as envs

from irp.envs.env import Env
from irp.agents.sarsa import Sarsa
from irp.agents.qlearning import Qlearning
from irp.wrappers.tiled import Tiled
from irp.wrappers.multi_sample import MultiSample

def eval(local_vars):
    results.append(irp.utils.evaluate(environments, local_vars['self'].policy, max_steps=10)[1])
    N = 50

    return not (np.mean(results[-N:]) > 0.95 and len(results[-N:]) == N)

results = []
callback = {'interval': 10, 'callback': eval}

image_parameters = {
    'subimage_width': 16,
    'subimage_height': 8,
    'overlap': 0.875
}
neighborhood_parameters = {
    'n_size': 1,
    'overlap': image_parameters['overlap'],
    'neighborhood': 'neumann'
}
tiling_parameters = {
    'tiles_per_dim': (4, 4, 2),
    'tilings': 64,
    'limits': [(0, 1), (0, 1), (0, 32)]
}
agent_parameters = {
    'alpha': 0.6,
    'max_t': 1000,
    'max_e': 3000,
    'ep_max': 1.0,
    'ep_min': 0.05,
    'ep_frac': 0.01,
    'gamma': 0.8,
}

(image, truth), (t_image, t_truth) = irp.utils.read_sample('case10_10.png'), irp.utils.read_sample('case10_11.png')
subimages, sublabels, t_subimages, t_sublabels = irp.utils.extract_subimages(
    image, truth, t_image, t_truth, **image_parameters
)

sample_coord = (192, 248)
# sample_coord = (272, 232)
# sample_coord = (208, 264)
# sample_coord = (304, 280)

sample_id = irp.utils.coord_to_id(sample_coord, image.shape, **image_parameters)

sample, label = subimages[sample_id], sublabels[sample_id]
t_sample, t_label = t_subimages[sample_id], t_sublabels[sample_id]
neighborhood = irp.utils.get_neighborhood_images(
    subimages, sublabels, sample_id, **dict(image_parameters, **neighborhood_parameters)
)

environments = MultiSample()

for sample, label in zip(*neighborhood):
    environment = Env(sample, label, n_thresholds=4)
    environment = Tiled(environment, **tiling_parameters)

    environments.add(environment)

irp.utils.show(*neighborhood[1])

# sample, label = list(zip(*neighborhood))[0]
# environment = Env(sample, label, n_thresholds=4)
# environment = Tiled(environment, **tiling_parameters)

t_environment = Env(t_sample, t_label, n_thresholds=4)
t_environment = Tiled(t_environment, **tiling_parameters)

agent = Sarsa(environments)
policy = agent.learn(callback=callback, **agent_parameters)

if agent.e < agent_parameters['max_e']:
    print('Finished within the maximum number of episodes;', agent.e)

N = 10
plt.plot(np.convolve(results, np.ones(N) / N, mode='same')); plt.show()

tis = []
done = False
state, info = t_environment.reset(ti=4)
tis.append(t_environment.ti)

for i in range(10):
    action = policy.predict(state)
    state, reward, done, info = t_environment.step(action)

    tis.append(t_environment.ti)

    if irp.utils.is_oscilating(tis) is True:
        break

print('Threshold history', tis)
print('Dissimilarity v.s. optimal dissimilarity', envs.utils.compute_dissimilarity(t_environment.label, t_environment.bitmask), t_environment.d_sim)

# if agent.e < agent_parameters['max_e']:
irp.utils.show(t_environment.bitmask, t_environment.label)