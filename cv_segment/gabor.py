# @Author:      HgS_1217_
# @Create Date: 2017/12/8

import cv2
import numpy as np


def build_filters():
    filters = []
    ksize = [7, 9, 11, 13, 15, 17]
    lamda = np.pi / 2.0
    for theta in np.arange(0, np.pi, np.pi / 4):
        for K in range(6):
            kern = cv2.getGaborKernel((ksize[K], ksize[K]), 1.0, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5 * kern.sum()
            filters.append(kern)
    return filters


def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum


def get_gabor(img):
    res = []
    filters = build_filters()
    image = cv2.GaussianBlur(img, (3, 3), 0)
    for i in range(len(filters)):
        res1 = process(image, filters[i])
        res.append(np.asarray(res1))

    return res


if __name__ == '__main__':
    for i in range(7, 8):
        img = cv2.imread("pics/test{}.jpg".format(i), cv2.IMREAD_GRAYSCALE)
        res = get_gabor(img)
        for j in range(len(res)):
            cv2.imwrite("pics/gabor/{}.jpg".format(j), res[j])
