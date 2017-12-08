# @Author:      HgS_1217_
# @Create Date: 2017/12/3

import cv2
import cv_segment.gabor as gabor


def canny(img):
    image = cv2.GaussianBlur(img, (7, 7), 0)
    image = gabor.get_gabor(image)[12]
    return cv2.Canny(image, 80, 100)


if __name__ == '__main__':
    for i in range(1, 9):
        img = cv2.imread("pics/test{}.jpg".format(i), cv2.IMREAD_GRAYSCALE)
        res = canny(img)
        cv2.imwrite("pics/canny{}.jpg".format(i), res)
