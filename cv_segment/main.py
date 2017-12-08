# @Author:      HgS_1217_
# @Create Date: 2017/11/26

import cv2
import numpy as np


def smooth_max(pxs):
    for i in range(1, len(pxs)-1):
        if pxs[i] - pxs[i-1] > 200 and pxs[i-1] > 0 or pxs[i] - pxs[i+1] > 200 and pxs[i+1] > 0:
            pxs[i] = 0
        elif pxs[i] - pxs[i-1] < -200 and pxs[i] - pxs[i+1] < -200:
            pxs[i] = int((pxs[i-1]+pxs[i+1])/2)
    return pxs


def search_valley(pxs):
    zeros, nonzs = 0, 0
    regions = []
    for i in range(len(pxs)):
        if pxs[i] == 0:
            if zeros == 0:
                if nonzs >= 5:
                    regions.append([1, i-nonzs, i-1])
                nonzs = 0
            zeros += 1
        else:
            if nonzs == 0:
                if zeros >= 5:
                    regions.append([0, i - zeros, i - 1])
                zeros = 0
            nonzs += 1
    if nonzs >= 5:
        regions.append([1, len(pxs) - nonzs, len(pxs) - 1])
    elif zeros >= 5:
        regions.append([0, len(pxs) - zeros, len(pxs) - 1])

    valids = [r[1:] for r in list(filter(lambda x: x[0] == 1, regions))]
    return valids


def local_ext(canny):
    borders = [np.where(row[:int(len(row)*2/3)] == 255) for row in canny]
    rightest_pxs = [np.max(b[0]) if len(b[0]) > 0 else 0 for b in borders]
    smooth_pxs = smooth_max(rightest_pxs)
    print(smooth_pxs)
    valids = search_valley(smooth_pxs)
    print(valids)
    valleys = []
    for valid in valids:
        r = smooth_pxs[valid[0]:valid[1]+1]
        max_indexs = np.where(r == np.max(r))[0]
        valleys.append(max_indexs[int(len(max_indexs)/2)] + valid[0])
    valley_pxs = [[v, rightest_pxs[v]] for v in valleys]
    print(valley_pxs)
    return valley_pxs


def get_roi(img):
    image = cv2.GaussianBlur(img, (5, 5), 0)
    canny = cv2.Canny(image, 80, 120)
    valley_pxs = local_ext(canny)
    for row, col in valley_pxs:
        cv2.circle(canny, (col, row), 5, (255, 0, 0), 3)
    return canny


if __name__ == '__main__':
    for i in range(5, 9):
        img = cv2.imread("pics/test{}.jpg".format(i), cv2.IMREAD_GRAYSCALE)
        res = get_roi(img)
        cv2.imwrite("pics/roi{}.jpg".format(i), res)
