

import cv2
import numpy as np
from matplotlib import pyplot as plt
def imshow(name, image, resize=1):
    H,W = image.shape[0], image.shape[1]
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image.astype(np.uint8))
    cv2.resizeWindow(name, round(resize*W), round(resize*H))
img = np.zeros((512,512,3),np.uint8)
cv2.line(img,(500,500),(2000,1001),(155,155,155),5)
imshow('brg',img)
cv2.waitKey(0)