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


def laplace(img):
    image = cv2.bilateralFilter(img, 9, 10, 10)
    # image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.medianBlur(image, 7)
    gray_lap = cv2.Laplacian(image, cv2.CV_16S, ksize=5)
    res = cv2.convertScaleAbs(gray_lap)
    _, res = cv2.threshold(res, 20, 255, cv2.THRESH_BINARY)
    res = cv2.GaussianBlur(res, (11, 11), 0)
    _, res = cv2.threshold(res, 150, 255, cv2.THRESH_BINARY)
    return res


if __name__ == '__main__':
    for i in range(1, 9):
        img = cv2.imread("pics/test{}.jpg".format(i), cv2.IMREAD_GRAYSCALE)
        # res = sobel(img)
        # cv2.imwrite("pics/sobel{}.jpg".format(i), res)
        res = laplace(img)
        res = cv2.medianBlur(res, 5)
        res = cv2.medianBlur(res, 11)
        cv2.imwrite("pics/laplace{}.jpg".format(i), res)
