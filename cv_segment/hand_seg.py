# @Author:      HgS_1217_
# @Create Date: 2017/12/26

import cv2
import numpy as np


def skin_ostu(img):
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    img_cr = cv2.split(img_ycrcb)[1]

    _, target = cv2.threshold(img_cr, 0, 255, cv2.THRESH_OTSU)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    target = cv2.morphologyEx(target, cv2.MORPH_OPEN, element)

    contours = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
    mymax = 0
    max_contours = None
    for cont in contours:
        area = cv2.contourArea(cont)
        if area > mymax:
            mymax = area
            max_contours = cont

    [mid_x, mid_y] = [int(x) for x in np.mean(max_contours, 0)[0]]
    mask = np.zeros(img.shape[:2], np.uint8)

    for [[y, x]] in max_contours:
        mask[x, y] = 255

    m = np.zeros((len(img)+2, len(img[0])+2), np.uint8)
    cv2.floodFill(mask, m, (mid_x, mid_y), (255, 255, 255))

    mask = (cv2.GaussianBlur(mask, (3, 3), 0)) / 255

    for i in range(len(img)):
        for j in range(len(img[0])):
            img[i, j] = img[i, j] * mask[i, j]

    return img
