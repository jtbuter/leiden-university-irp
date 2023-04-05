import numpy as np
import nrrd
import cv2

for patient_id in [9, 10, 12]:
    image_name = f"../data/trus/Case{patient_id}_US_resampled.nrrd"
    label_name = f"../data/trus/Case{patient_id}_US_resampled-label.nrrd"

    imagedata, header = nrrd.read(image_name, index_order = 'C')
    labeldata, header = nrrd.read(label_name, index_order = 'C')

    for i in range(header['sizes'][-1]):
        image = np.asarray(imagedata[i, :, :], dtype = np.uint8)

        print(np.unique(image))

        label = np.asarray(labeldata[i, :, :], dtype = np.uint8) * 255

        prostate = np.max(label) != 0
        visible = np.max(image) > 0

        if prostate and visible:
            cv2.imwrite(f'../data/trus/images/case{patient_id}_{i}.png', image)
            cv2.imwrite(f'../data/trus/labels/case{patient_id}_{i}.png', label)
