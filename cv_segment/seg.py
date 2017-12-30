# @Author:      HgS_1217_
# @Create Date: 2017/12/19

# @Author:      HgS_1217_
# @Create Date: 2017/12/3

import cv2
import os
import numpy as np
from cv_segment.utils import resize
from config import INPUT_DIR, OUTPUT_DIR, RESIZE_DIR, CON_RESIZE_DIR, CON_OUTPUT_DIR, TEST_DIR, CON_INPUT_DIR


def laplace(img):
    res = cv2.bilateralFilter(img, 9, 10, 10)
    res = cv2.medianBlur(res, 7)
    gray_lap = cv2.Laplacian(res, cv2.CV_16S, ksize=5)
    res = cv2.convertScaleAbs(gray_lap)
    _, res = cv2.threshold(res, 45, 255, cv2.THRESH_BINARY)
    res = cv2.GaussianBlur(res, (3, 3), 0)
    _, res = cv2.threshold(res, 150, 255, cv2.THRESH_BINARY)

    res = cv2.medianBlur(res, 13)
    res = cv2.GaussianBlur(res, (15, 15), 0)
    _, res = cv2.threshold(res, 150, 255, cv2.THRESH_BINARY)
    res = cv2.GaussianBlur(res, (15, 15), 0)
    return res


def canny(img):
    res = cv2.bilateralFilter(img, 9, 10, 10)
    res = cv2.medianBlur(res, 3)
    res = cv2.Canny(res, 42, 95)
    res = cv2.GaussianBlur(res, (15, 15), 0)
    _, res = cv2.threshold(res, 25, 255, cv2.THRESH_BINARY)

    res = cv2.medianBlur(res, 9)
    res = cv2.GaussianBlur(res, (15, 15), 0)
    _, res = cv2.threshold(res, 75, 255, cv2.THRESH_BINARY)
    res = cv2.GaussianBlur(res, (15, 15), 0)
    return res


def seg(img):
    """
    Preprocess the dataset image to mark the palmprint region
    """
    c = canny(img)
    l = laplace(img)
    res = c / 2 + l / 2
    _, res = cv2.threshold(res, 125, 255, cv2.THRESH_BINARY)
    return res


def local():
    for i in range(4, 9):
        img = cv2.imread("pics/test{}.jpg".format(i), cv2.IMREAD_GRAYSCALE)
        con = contrast(img)
        res = seg(con)
        cv2.imwrite("pics/canny{}.jpg".format(i), res)


def smaller(image):
    width, height = len(image[0]), len(image)
    return image[int(height / 4):int(height / 4 * 3), int(width / 4):int(width / 4 * 3)]


def keep_ori():
    for parent, dirnames, filenames in os.walk(RESIZE_DIR):
        if len(filenames) > 0:
            print("DIR NOT EMPTY!")
            return
    for parent, dirnames, filenames in os.walk(TEST_DIR):
        for filename in filenames:
            if filename.split(".")[-1] == "jpg":
                total_name = os.path.join(parent, filename).replace("\\", "/")
                print(total_name)
                img = cv2.imread(total_name, cv2.IMREAD_GRAYSCALE)
                resize_raw = resize(img, 256, 256)
                cv2.imwrite("%s/%s" % (RESIZE_DIR, filename), smaller(resize_raw))


def contrast(img):
    mean = np.mean(img)
    res = np.array(img, dtype=np.int16)
    res = 3 * res - 2.8 * mean - 50
    res = np.maximum(res, np.zeros(res.shape))
    res = np.minimum(res, np.ones(res.shape) * 255)
    res = np.array(res, dtype=np.uint8)
    return res


def main():
    for parent, dirnames, filenames in os.walk(CON_OUTPUT_DIR):
        if len(filenames) > 0:
            print("DIR NOT EMPTY!")
            return
    for parent, dirnames, filenames in os.walk(INPUT_DIR):
        for filename in filenames:
            if filename.split(".")[-1] == "jpg":
                total_name = os.path.join(parent, filename).replace("\\", "/")
                print(total_name)
                img = cv2.imread(total_name, cv2.IMREAD_GRAYSCALE)
                con = contrast(img)
                res = seg(con)
                resize_raw = resize(con, 256, 256)
                _, resize_output = cv2.threshold(resize(res, 256, 256), 125, 255, cv2.THRESH_BINARY)
                cv2.imwrite("%s/%s" % (CON_OUTPUT_DIR, filename), smaller(resize_output))
                cv2.imwrite("%s/%s" % (CON_RESIZE_DIR, filename), smaller(resize_raw))


if __name__ == '__main__':
    main()
    # keep_ori()
    # local()
