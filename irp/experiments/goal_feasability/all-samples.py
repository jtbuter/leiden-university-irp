import matplotlib.pyplot as plt
import numpy as np
from irp.wrappers import Discretize, MultiSample
import irp.utils
import irp.envs
from gym.wrappers import TimeLimit
from irp.experiments.goal_feasability.env import Env
import irp.experiments.goal_feasability.q as q

shape = (512, 512)
subimage_width, subimage_height = 16, 8
overlap = 0.75
n_size = 2
n_thresholds = 6
bins = (3, 3, 4)
params = {
    'episodes': 5000, 'alpha': 0.3, 'gamma': 0.9,
    'epsilon': 1.0, 'epsilon_decay': 0.0025, 'min_eps': 0.05, 'learn_delay': 1000
}
coords = irp.utils.extract_subimages(np.zeros((512, 512)), subimage_width, subimage_height, 0)[1]

train_subimages, train_sublabels = np.asarray(irp.utils.make_sample_label(
    'case10_10.png', width=subimage_width, height=subimage_height, overlap=overlap, idx=None
)[0])
test_subimages, test_sublabels = np.asarray(irp.utils.make_sample_label(
    'case10_11.png', width=subimage_width, height=subimage_height, overlap=overlap, idx=None
)[0])

result = np.zeros((512, 512))
result2 = np.zeros((512, 512))

for coord in coords:
    x, y = coord

    # Make sure we're not wasting processing power right now
    if not (x >= 192 and x <= 336 and y >= 176 and y <= 288):
        continue

    print("Processing coord", coord)

    train_neighborhood_subimages = []
    train_neighborhood_sublabels = []
    neighborhood_coords = irp.utils.get_neighborhood(
        coord, (512, 512), subimage_width, subimage_height, overlap=overlap, n_size=n_size
    )

    # Make sure all the subimages can actually be solved
    for neighbor in neighborhood_coords:
        id = irp.utils.coord_to_id(neighbor, (512, 512), subimage_width, subimage_height, overlap=overlap)

        subimage = train_subimages[id]
        sublabel = train_sublabels[id]

        mini, maxi = np.min(subimage), np.max(subimage)

        tis = np.linspace(mini, maxi, n_thresholds, dtype=np.uint8).tolist()
        tis = np.concatenate(([mini - 1], tis))

        best_dissim = np.inf
        best_bitmask = -1

        for ti in tis:
            bitmask = irp.envs.utils.apply_threshold(subimage, ti)
            dissim = irp.envs.utils.compute_dissimilarity(bitmask, sublabel)

            if dissim < best_dissim:
                best_dissim = dissim
                best_bitmask = bitmask

        if best_dissim <= 0.08:
            train_neighborhood_subimages.append(subimage)
            train_neighborhood_sublabels.append(sublabel)

    print("States used for learning:", len(train_neighborhood_subimages))

    envs = [
        TimeLimit(Discretize(Env(sample, label, n_thresholds), [0, 0, 1], [1, 1, bins[2]], bins), 30)
        for sample, label in zip(train_neighborhood_subimages, train_neighborhood_sublabels)
    ]

    env: Env = MultiSample(envs)

    qtable = q.learn(env, **params)

    # Set-up testing
    id = irp.utils.coord_to_id(coord, (512, 512), subimage_width, subimage_height, overlap=0)
    test_subimage, test_sublabel = test_subimages[id], test_sublabels[id]
    env = Discretize(Env(test_subimage, test_sublabel, n_thresholds), [0, 0, 1], [1, 1, bins[2]], bins)

    s = env.reset(threshold_i=0)

    for i in range(15):
        a = np.argmax(qtable[tuple(s)])
        s, r, d, i = env.step(a)

    print(d)

    threshold_i = env.threshold_i
    intensity = env.intensities[threshold_i]
    bit_mask = irp.envs.utils.apply_threshold(test_subimage, intensity)

    result[y:y+subimage_height, x:x+subimage_width] = bit_mask
    result2[y:y+subimage_height, x:x+subimage_width] = test_sublabel

# plt.imshow(result2, cmap='gray', vmin=0, vmax=1)
plt.imshow(result, cmap='gray', vmin=0, vmax=1)
plt.show()


