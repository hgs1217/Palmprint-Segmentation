# @Author:      HgS_1217_
# @Create Date: 2017/12/3

import cv2


# maybe not correct
def rotate(image):
    height, width = len(image), len(image[0])
    rot = cv2.getRotationMatrix2D((width / 2, height / 2), 90, 1)
    rot[0, 2] += (height - width) / 2
    rot[1, 2] += (width - height) / 2
    return cv2.warpAffine(image, rot, (height, width), borderValue=(255, 255, 255))


def resize(image, x, y):
    width, height = len(image[0]), len(image)
    img = image
    if width > height:
        img = image[:, int(width / 2 - height / 2):int(width / 2 + height / 2)]
    elif width < height:
        img = image[int(height / 2 - width / 2):int(height / 2 + width / 2), :]

    return cv2.resize(img, (x, y), interpolation=cv2.INTER_CUBIC)
