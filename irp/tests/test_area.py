import cv2
import numpy as np

a = np.zeros((32, 16), dtype=np.uint8)
a[2:-2, 2:-2] = 1
b = a.copy()
b[4:-4, 4:-4] = 0

cnt_a = cv2.findContours(a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0]
cnt_b = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0]

assert cv2.contourArea(cnt_a) == cv2.contourArea(cnt_b)