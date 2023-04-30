from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from irp.experiments.goal_feasability.env import Env
import irp.experiments.goal_feasability.q as q
import irp.utils
import irp.envs
import numpy as np
from irp.wrappers import Discretize
from irp.envs.ultrasound.ultra_sound_env import UltraSoundEnv
import matplotlib.pyplot as plt
import cv2

width, height = 16, 8

for j in range(5, 17):
    subimages, sublabels = np.asarray(irp.utils.get_subimages(f'case10_{j}.png', width=width, height=height))

    # 1889, 1890
    # 1953, 1954

    n_thresholds = 15
    best_intensities = np.zeros((int(512 / height), int(512 / width)))

    for i, (subimage, sublabel) in enumerate(zip(subimages, sublabels)):
        if np.max(sublabel) == 0:
            continue

        intensities = np.linspace(np.min(subimage), np.max(subimage), n_thresholds, dtype=np.uint8).tolist()
        best_dissim = np.inf
        best_intensity = -1

        for threshold_i in range(15):
            for size in [0, 2, 5]:
                intensity = intensities[threshold_i]
                bit_mask = irp.envs.utils.apply_threshold(subimage, intensity)
                bit_mask = irp.envs.utils.apply_opening(bit_mask, size)
                dissim = irp.envs.utils.compute_dissimilarity(bit_mask, sublabel)

                if dissim < best_dissim:
                    best_dissim = dissim
                    best_intensity = intensity

        best_intensities[np.unravel_index(i, best_intensities.shape)] = best_dissim

    print(
        round(np.mean(best_intensities[best_intensities > 0]), 3), round(np.std(best_intensities[best_intensities > 0]), 3)
    )

    ax = plt.subplot()
    im = ax.imshow(best_intensities)

    # create an Axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)

    plt.show()
