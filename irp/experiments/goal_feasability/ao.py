import sklearn.metrics
import numpy as np
import irp.utils
import matplotlib.pyplot as plt

def recall(label, bitmask):
    tp = ((label == 255) & (bitmask == 255)).sum()
    fn = ((bitmask == 0) & (label != bitmask)).sum()

    if tp + fn == 0:
        return 0.0

    return tp / (tp + fn)

for i in range(30):
    size = 300

    a, b = sorted(np.random.choice(range(size + 1), 2))
    c, d = sorted(np.random.choice(range(size + 1), 2))

    label = np.zeros((size, size))
    label[a:b, c:d] = 255

    a, b = sorted(np.random.choice(range(size + 1), 2))
    c, d = sorted(np.random.choice(range(size + 1), 2))

    bitmask = np.zeros((size, size))
    bitmask[a:b, c:d] = 255

    sk = sklearn.metrics.recall_score((label / 255).flatten(), (bitmask / 255).flatten(), zero_division=0)
    ours = recall(label, bitmask)

    plt.title(f'{round(sk, 3)} {round(ours, 3)}')
    irp.utils.show(np.logical_xor(label, bitmask) * 255)

    assert np.isclose(sk, ours), 'failed'
