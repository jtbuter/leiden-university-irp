import itertools
from matplotlib import pyplot as plt
import numpy as np

import irp
import irp.utils
import irp.envs as envs
import irp.wrappers as wrappers
from irp.envs.ultrasound.ultra_sound_env import UltraSoundEnv

coords = [(256, 176), (288, 176), (240, 184), (256, 184), (224, 192), (240, 192), (304, 192), (224, 200), (320, 200), (224, 208), (320, 208), (208, 216), (208, 224), (336, 224), (208, 232), (192, 240), (336, 240), (192, 248), (304, 248), (208, 264), (256, 264), (272, 264), (208, 272), (224, 272), (240, 272), (256, 272), (288, 272), (240, 280), (288, 280), (304, 280), (304, 288), (320, 288)]

train_name = 'case10_10.png'
test_name = 'case10_11.png'
s_width, s_height = 16, 8
overlap = 0
n_thresholds = 4

# Get all the subimages
xs, ys = irp.utils.get_subimages(train_name, s_width, s_height, overlap)
real = irp.utils.read_image(train_name, add_root=True)[1]

coords = irp.utils.extract_subimages(real, s_width, s_height, overlap)[1]
pred = np.zeros((512, 512))

for coord in coords:
    x, y = coord
    number = irp.utils.coord_to_id(coord, (512, 512), s_width, s_height, overlap)
    sample, label = xs[number], ys[number]

    ths = envs.utils.get_intensity_spectrum(sample, n_thresholds, add_minus=True)
    d, seq = irp.utils.get_best_dissimilarity(sample, label,
        actions=[ths, [0, 2, 5]], fns=[envs.utils.apply_threshold, envs.utils.apply_opening], return_seq=True
    )

    bitmask = envs.utils.apply_threshold(sample, *seq[0])
    bitmask = envs.utils.apply_opening(bitmask, *seq[1])

    pred[y:y+s_height, x:x+s_width] = bitmask
    
plt.title(irp.utils.f1(real, pred))
irp.utils.show(pred)
