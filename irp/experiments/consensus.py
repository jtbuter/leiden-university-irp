from typing import Tuple
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import irp.utils

def sliding_window(source, s_width, s_height, step_x, step_y):
    view = sliding_window_view(source, (s_height, s_width))[::step_y, ::step_x]
    num_elements = view.shape[0] * view.shape[1]

    return view.reshape((num_elements, view.shape[2], view.shape[3]))

def normalize_coord(main_coord: Tuple[int, int], neighbor_coord: Tuple[int, int], x_step: int, y_step: int):
    return neighbor_coord[0] - main_coord[0] + x_step, neighbor_coord[1] - main_coord[1] + y_step

# image = np.array([
#     [0, 0, 0, 0, 0],
#     [0, 1, 1, 1, 0],
#     [0, 1, 1, 1, 0],
#     [0, 1, 1, 1, 0],
#     [0, 0, 0, 0, 0]
# ], dtype=np.uint8)

image = np.ones((512, 512), dtype=np.uint8)
s_width, s_height, x_step, y_step = 16, 8, 8, 4

samples = sliding_window(image, s_width, s_height, x_step, y_step)
sample_coord = (256, 232)
coords = irp.utils.get_neighborhood(sample_coord, image.shape, s_width, s_height, (x_step, y_step), n_size=1, neighborhood='neumann')
ids = list(irp.utils.coord_to_id(coord, image.shape, s_width, s_height, (x_step, y_step)) for coord in coords)

result = np.zeros((2 * y_step + s_height, 2 * x_step + s_width))

for coord, sample_id in zip(coords, ids):
    normalized_coord = normalize_coord(sample_coord, coord, x_step, y_step)
    x, y = normalized_coord

    result[y:y+s_height,x:x+s_width] += samples[sample_id]
    # result[y:y+s_height,x:x+s_width] = samples[sample_id]

print(np.max(result[y:y+s_height,x:x+s_width]))

x, y = sample_coord

irp.utils.show(result)
# irp.utils.show(result[y:y+s_height, x:x+s_width])
