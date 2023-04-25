import irp.envs.multi_sample.multi_sample

# # - Percentage states dat al voldoet aan het criterium zonder dat we
# #   enige acties hebben uitgevoerd.
# # - 

# import os
# import cv2
# import matplotlib.pyplot as plt
# import numpy as np

# import irp
# from irp import utils

# def get_subimages(filename):
#     # Define the paths to the related parent directories
#     base_path = os.path.join(irp.GIT_DIR, "../data/trus/")
#     image_path = os.path.join(base_path, 'images')
#     label_path = os.path.join(base_path, 'labels')
#     # Read the image and label
#     image = utils.read_image(os.path.join(image_path, filename))
#     image = utils.median_filter(image, 7)
#     label = utils.read_image(os.path.join(label_path, filename))

#     subimages = utils.extract_subimages(image, 32, 16)[0]
#     sublabels = utils.extract_subimages(label, 32, 16)[0]

#     return subimages, sublabels

# filename = 'case10_10.png'
# subimages, sublabels = get_subimages(filename)
# num_thresholds = 15

# total = 0

# for z, (x, y) in enumerate(zip(subimages, sublabels)):
#     tis = np.linspace(np.min(x), np.max(x), num_thresholds, dtype=np.uint8).tolist()
#     print(tis)

#     # for i in range(0, num_thresholds):
#     #     for j in range(i, num_thresholds):
#     #         ti_l, ti_r = tis[i], tis[j]

#     #         bit_mask = utils.apply_threshold(x, ti_l, ti_r)

#     #         plt.title(f'{ti_l} {ti_r}')
#     #         plt.imshow(bit_mask, cmap='gray', vmin=0, vmax=1)
#     #         plt.show()

#     # break
