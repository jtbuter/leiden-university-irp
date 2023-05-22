import numpy as np
import matplotlib.pyplot as plt

from typing import List, Optional, Tuple, Union

def diamond(n):
    a = np.arange(n)
    b = np.minimum(a, a[::-1])
    
    return (b[:, None] + b) >= (n - 1) // 2

def get_neighborhood(
    coord: Union[int, Tuple],
    shape: Tuple[int, int],
    subimage_width: int,
    subimage_height: int,
    overlap: Optional[float] = 0,
    n_size: Optional[int] = 1,
    neighborhood = 'moore'
) -> List[Tuple]:
    width_step_size = round((1 - overlap) * subimage_width, 0)
    height_step_size = round((1 - overlap) * subimage_height, 0)

    x, y = coord
    coords = []

    if neighborhood == 'neumann':
        neighbor_map = diamond(n_size * 2 + 1).flatten()
    else:
        neighbor_map = np.ones((n_size * 2 + 1, n_size * 2 + 1), dtype=bool).flatten()

    for y_i in range(-n_size, n_size + 1):
        y_i *= height_step_size

        for x_i in range(-n_size, n_size + 1):
            x_i *= width_step_size

            coords.append((x + x_i, y + y_i))

    return np.asarray(coords)[neighbor_map]

coords = get_neighborhood((256, 224), shape=(512, 512), subimage_width=16, subimage_height=8, neighborhood='moore', n_size=4)
result = np.zeros((512, 512))

for (x, y) in coords:
    result[y:y+8, x:x+16] = 255

plt.imshow(result); plt.show()
