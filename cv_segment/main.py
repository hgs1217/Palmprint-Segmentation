# @Author:      HgS_1217_
# @Create Date: 2017/11/26

import cv2
import math
import numpy as np


def sobel_segmentation(image):
    x = cv2.Sobel(image, cv2.CV_16S, 1, 0, ksize=5)
    y = cv2.Sobel(image, cv2.CV_16S, 0, 1, ksize=5)

    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)

    return cv2.addWeighted(absX, 0.5, absY, 0.5, 0)


def rotate(image):
    height, width = len(image), len(image[0])
    rot = cv2.getRotationMatrix2D((width / 2, height / 2), 90, 1)
    rot[0, 2] += (height - width) / 2
    rot[1, 2] += (width - height) / 2
    return cv2.warpAffine(image, rot, (height, width), borderValue=(255, 255, 255))


def resize(image):
    return cv2.resize(image, (180, 320), interpolation=cv2.INTER_CUBIC)


def main():
    for i in range(1, 6):
        img = cv2.imread("../test{}.jpg".format(i), cv2.IMREAD_GRAYSCALE)
        image = cv2.bilateralFilter(img, 9, 75, 75)
        sobel = sobel_segmentation(image)

        _, res = cv2.threshold(sobel, 50, 255, cv2.THRESH_TOZERO)
        # res = resize(res)

        cv2.imwrite("result{}.jpg".format(i), res)


if __name__ == '__main__':
    main()
