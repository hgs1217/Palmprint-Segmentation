# @Author:      HgS_1217_
# @Create Date: 2017/12/3

import cv2
import numpy as np
import math


def ang_to_rad(angle):
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


def resize_for_roi(roi, src_w, src_h, dst_w, dst_h):
    actual_h, actual_w = dst_h, dst_w
    if dst_w > dst_h:
        actual_w = -int(dst_w / 2 - dst_h / 2) + int(dst_w / 2 + dst_h / 2)
    elif dst_w < dst_h:
        actual_h = -int(dst_h / 2 - dst_w / 2) + int(dst_h / 2 + dst_w / 2)
    resized_actual = cv2.resize(roi, (actual_w, actual_h), interpolation=cv2.INTER_CUBIC)
    result = np.zeros((dst_h, dst_w))

    if dst_w > dst_h:
        result[:, int(dst_w / 2 - dst_h / 2):int(dst_w / 2 + dst_h / 2)] = resized_actual
    elif dst_w < dst_h:
        result[int(dst_h / 2 - dst_w / 2):int(dst_h / 2 + dst_w / 2), :] = resized_actual
    else:
        result = resized_actual
    return result


def enhance_contrast(image):
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    return cv2.LUT(image, cdf)


def rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(image, M, (nW, nH))


def sobel_segmentation(image):
    x = cv2.Sobel(image, cv2.CV_16S, 1, 0, ksize=5)
    y = cv2.Sobel(image, cv2.CV_16S, 0, 1, ksize=5)

    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)

    return cv2.addWeighted(absX, 0.5, absY, 0.5, 0)


def sobel(img):
    image = cv2.bilateralFilter(img, 7, 75, 75)
    image = cv2.medianBlur(image, 7)
    sobel = sobel_segmentation(image)
    _, res = cv2.threshold(sobel, 50, 255, cv2.THRESH_TOZERO)
    return res
