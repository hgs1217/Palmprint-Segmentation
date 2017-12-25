# @Author:      HgS_1217_
# @Create Date: 2017/12/3

import cv2
import numpy as np
import math


def angToRad(angle):
    return angle / 180 * math.pi


def rotate(image, angle):
    height, width = image.shape[:2]
    rot = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    return cv2.warpAffine(image, rot, (width, height))


def resize(image, x, y):
    width, height = len(image[0]), len(image)
    img = image
    if width > height:
        img = image[:, int(width / 2 - height / 2):int(width / 2 + height / 2)]
    elif width < height:
        img = image[int(height / 2 - width / 2):int(height / 2 + width / 2), :]

    return cv2.resize(img, (x, y), interpolation=cv2.INTER_CUBIC)


def enhance_contrast(image):
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    return cv2.LUT(image, cdf)
