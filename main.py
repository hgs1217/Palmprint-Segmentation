# @Author:      HgS_1217_
# @Create Date: 2017/12/22

import numpy as np
import cv2
from cv_segment.hand_seg import skin_ostu
from cv_segment.extract_roi import get_roi, mapping
from cv_segment.seg import contrast
from cnn.segnet import SegNet

IMG_PATH = "D:/Computer Science/Github/Palmprint-Segmentation/cv_segment/pics/testt.jpg"
OUT_PATH = "D:/Computer Science/Github/Palmprint-Segmentation/cv_segment/pics/net_res4.jpg"


def main():
    img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
    raw = np.array(img)
    ostu = skin_ostu(img)
    ostu_gray = cv2.cvtColor(ostu, cv2.COLOR_BGR2GRAY)
    roi_for_cnn, roi, show_roi, angle, cut_range, need_flip, cut_seg = get_roi(ostu_gray, ostu_gray, 0)
    roi_for_cnn = contrast(roi_for_cnn)

    res_seg = SegNet(input_size=128).check(roi_for_cnn)

    res_seg = np.array(res_seg, dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    res = cv2.erode(res_seg, kernel)

    final = mapping(raw, res, angle, cut_range, need_flip)
    cv2.imwrite(OUT_PATH, final)


if __name__ == '__main__':
    main()
