# @Author:      HgS_1217_
# @Create Date: 2017/12/3

import cv2


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


if __name__ == '__main__':
    for i in range(1, 9):
        img = cv2.imread("pics/test{}.jpg".format(i), cv2.IMREAD_GRAYSCALE)
        res = sobel(img)
        cv2.imwrite("pics/result{}.jpg".format(i), res)
