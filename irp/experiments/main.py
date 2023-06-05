import itertools
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
    results.append(irp.utils.evaluate(t_environment, local_vars['self'].policy, max_steps=10)[0])
    
    return True

    N = 50

    return not (np.mean(results[-N:]) > 0.95 and len(results[-N:]) == N)

callback = {'interval': 10, 'callback': eval}

image_parameters = {
    'subimage_width': 32,
    'subimage_height': 16,
    'overlap': 0.5
}
neighborhood_parameters = {
    'n_size': 0,
    'overlap': image_parameters['overlap'],
    'neighborhood': 'neumann'
}
tiling_parameters = {
    'tiles_per_dim': (2, 2, 2),
    'tilings': 64,
    'limits': [(0, 1), (0, 1), (0, 4)]
}
agent_parameters = {
    'alpha': 0.6,
    'max_t': 1000,
    'max_e': 3000,
    'ep_max': 0.6,
    'ep_min': 0.6,
    'ep_frac': 0.01,
    'gamma': 0.8,
}
environment_parameters = {
    'n_thresholds': 6,
    'opening': 8
}

(image, truth), (t_image, t_truth) = irp.utils.read_sample('case10_10.png'), irp.utils.read_sample('case10_11.png')
subimages, sublabels, t_subimages, t_sublabels = irp.utils.extract_subimages(
    image, truth, t_image, t_truth, **image_parameters
)

result = np.zeros((512, 512))
coords = irp.utils.extract_coordinates(image.shape, **dict(image_parameters, **{'overlap': 0}))

for coord in coords:
    for x_p, y_p in [(0, -2), (-4, 0), (0, 0), (4, 0), (0, 2)]:
        results = []
        orig_sample_coord = (192, 160)
        sample_coord = (orig_sample_coord[0] + x_p, orig_sample_coord[1] + y_p) # +- 8 +- 4, orig. (192, 248)
        sample_id = irp.utils.coord_to_id(sample_coord, image.shape, **image_parameters)

        print(irp.utils.id_to_coord(sample_id, image.shape, **image_parameters))

        sample, label = subimages[sample_id], sublabels[sample_id]
        t_sample, t_label = t_subimages[sample_id], t_sublabels[sample_id]
        neighborhood = irp.utils.get_neighborhood_images(
            subimages, sublabels, sample_id, **dict(image_parameters, **neighborhood_parameters)
        )

        sample, label = list(zip(*neighborhood))[0]
        # irp.utils.show(label)
        environment = Env(sample, label, **environment_parameters)
        environment = Tiled(environment, **tiling_parameters)

        t_environment = Env(t_sample, t_label, **environment_parameters)
        t_environment = Tiled(t_environment, **tiling_parameters)

        agent = Qlearning(environment)
        policy = agent.learn(callback=callback, **agent_parameters)

        if agent.e < agent_parameters['max_e']:
            print('Finished within the maximum number of episodes;', agent.e)

        # N = 10
        # plt.plot(np.convolve(results, np.ones(N) / N, mode='same')); plt.show()

        tis = []
        done = False
        state, info = t_environment.reset(ti=4)
        tis.append(t_environment.ti)
        bitmask = t_environment.bitmask.copy()

        for i in range(10):
            action = policy.predict(state)
            state, reward, done, info = t_environment.step(action)

            tis.append(t_environment.ti)

            if irp.utils.is_oscilating(tis) is True or tis[-1] == tis[-2]:
                break

            bitmask = t_environment.bitmask.copy()

        print('Threshold history', tis)
        print('Dissimilarity v.s. optimal dissimilarity', envs.utils.compute_dissimilarity(t_environment.label, bitmask), t_environment.d_sim)

        best_sequence = irp.utils.get_best_dissimilarity(t_sample, t_label, [t_environment.intensity_spectrum, [8]], [envs.utils.apply_threshold, envs.utils.apply_opening], return_seq=True)[1]
        best_bitmask = irp.utils.apply_action_sequence(t_sample, best_sequence, [envs.utils.apply_threshold, envs.utils.apply_opening])

        irp.utils.show(bitmask, best_bitmask, t_label)
