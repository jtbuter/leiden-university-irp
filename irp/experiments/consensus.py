import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import irp.utils

def sliding_window(source, s_width, s_height, step_x, step_y):
    view = sliding_window_view(source, (s_height, s_width))[::step_y, ::step_x]
    num_subs = view.shape[0] * view.shape[1]

    return view.reshape((num_subs, view.shape[2], view.shape[3]))

image = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
])

s_width, s_height, x_step, y_step = 2, 2, 1, 1

coords = irp.utils.get_neighborhood((1, 1), image.shape, s_width, s_height, (x_step, y_step), 1, neighborhood='neumann')

print(coords)