import os
from irp import GIT_DIR, ROOT_DIR
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

for width, height in [(16, 8)]:
    print('\n', width, height)

    for j in range(10, 12):
        label = irp.utils.read_image(os.path.join(GIT_DIR, f'../data/trus/labels/case10_{j}.png'))
        subimages, sublabels = np.asarray(irp.utils.get_subimages(f'case10_{j}.png', width=width, height=height))

        n_thresholds = 6
        best_intensities = np.zeros((512, 512))
        best_solvables = np.zeros((512, 512))
        best_comps = np.zeros((512, 512))
        best_areas = np.zeros((512, 512))
        best_objects = np.zeros((512, 512))
        difficulty = np.zeros((512, 512))
        mean_intensities = []

        for i, (subimage, sublabel) in enumerate(zip(subimages, sublabels)):
            if np.max(sublabel) == 0:
                continue

            intensities = np.linspace(np.min(subimage), np.max(subimage), n_thresholds, dtype=np.uint8).tolist()
            best_dissim = np.inf
            best_intensity, best_area, best_comp = -1, -1, -1

            mean_intensities.append(len(set(intensities)))

            for threshold_i in range(n_thresholds):
                # for size in [0, 2, 5]:
                for size in [0]:
                    intensity = intensities[threshold_i]
                    bit_mask = irp.envs.utils.apply_threshold(subimage, intensity)
                    bit_mask = irp.envs.utils.apply_opening(bit_mask, size)
                    area, comp, obj = UltraSoundEnv.observation(bit_mask)
                    dissim = irp.envs.utils.compute_dissimilarity(bit_mask, sublabel)

                    if dissim < best_dissim:
                        best_dissim = dissim
                        best_intensity = intensity
                        best_area = area
                        best_comp = comp
                        best_obj = obj

            # best_intensities[i * height, i * width] = best_dissim
            x, y = irp.utils.unravel_index(i, width, height, 512)
            best_intensities[y:y+height, x:x+width] = best_dissim
            best_solvables[y:y+height, x:x+width] = float(best_dissim) <= 0.08
            best_comps[y:y+height, x:x+width] = best_comp
            best_areas[y:y+height, x:x+width] = best_area
            best_objects[y:y+height, x:x+width] = best_obj
            difficulty[y:y+height, x:x+width] = float(best_dissim) >= 0.02

            if float(best_dissim) >= 0.03:
                print(x, y)

            # print(j, np.mean(mean_intensities), np.std(mean_intensities))

        print(
            round(np.sum(best_intensities == True) / np.sum(label == 255), 3),
        )
        # print(
        #     round(np.mean(best_intensities[best_intensities > 0]), 3),
        #     round(np.std(best_intensities[best_intensities > 0]), 3)
        # )

        # plt.figure(figsize=(15,15))
        fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(15, 15))

        axes = axes.flatten()

        for ax in axes:
            ax.imshow(label, cmap='gray')

        ims = [
            axes[0].imshow(best_intensities, alpha=0.8),
            axes[1].imshow(best_solvables, alpha=0.8),
            axes[2].imshow(best_comps, alpha=0.8),
            axes[3].imshow(best_areas, alpha=0.8),
            axes[4].imshow(best_objects, alpha=0.8),
            axes[5].imshow(difficulty, alpha=0.8)
        ]

        axes[0].title.set_text('Dissimilarity')
        axes[1].title.set_text('Solvable')
        axes[2].title.set_text('Compactness')
        axes[3].title.set_text('Area')
        axes[4].title.set_text('Objects')
        axes[5].title.set_text('Difficult')

        for im, ax in zip(ims, axes):
            # create an Axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)

            cbar = plt.colorbar(im, cax=cax)
            cbar.solids.set(alpha=1)

            ax.set_xlim([180, 360])
            ax.set_ylim([300, 150])
        # fig.tight_layout()
        plt.show()
