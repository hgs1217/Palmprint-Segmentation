# @Author:      HgS_1217_
# @Create Date: 2017/12/22

import tensorflow as tf
import numpy as np
import random
import cv2
import os
from config import OUTPUT_DIR, RESIZE_DIR
from cv_segment.hand_seg import skin_ostu
from cv_segment.extract_roi import get_roi, mapping
from cv_segment.seg import contrast
from cnn.segnet import SegNet
from cv_segment.utils import resize

IMG_PATH = "D:/Computer Science/Github/Palmprint-Segmentation/cv_segment/pics/test0.jpg"
OUT_PATH = "D:/Computer Science/Github/Palmprint-Segmentation/cv_segment/pics/net_res2.jpg"


def main():
    img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
    ostu = skin_ostu(img)
    cv2.imwrite("D:/Computer Science/Github/Palmprint-Segmentation/cv_segment/pics/r1.jpg", ostu)
    ostu_gray = cv2.cvtColor(ostu, cv2.COLOR_BGR2GRAY)
    roi_for_cnn, roi, show_roi, angle, cut_range, need_flip, cut_seg = get_roi(ostu_gray, ostu_gray, 0)
    roi_for_cnn = contrast(roi_for_cnn)
    cv2.imwrite("D:/Computer Science/Github/Palmprint-Segmentation/cv_segment/pics/r2.jpg", roi_for_cnn)
    cv2.imwrite("D:/Computer Science/Github/Palmprint-Segmentation/cv_segment/pics/r2_5.jpg", show_roi)

    res_seg = SegNet(input_size=128).check(roi_for_cnn)

    res_seg = np.array(res_seg, dtype=np.uint8)
    res_seg = cv2.GaussianBlur(res_seg, (5, 5), 0)
    _, res_seg = cv2.threshold(res_seg, 175, 255, cv2.THRESH_BINARY)
    cv2.imwrite("D:/Computer Science/Github/Palmprint-Segmentation/cv_segment/pics/r3.jpg", res_seg)

    final = mapping(img, res_seg, angle, cut_range, need_flip)
    cv2.imwrite(OUT_PATH, final)


if __name__ == '__main__':
    main()
