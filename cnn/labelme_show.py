# @Author:      HgS_1217_
# @Create Date: 2017/12/25

import numpy as np
import tensorflow as tf
from cv_segment.utils import rotate
import cv2


IMG_PATH = "D:/Computer Science/dataset/palmprint/con_out/0005_m_l_01.jpg"


a = cv2.imread("../cv_segment/pics/test4.jpg", cv2.IMREAD_GRAYSCALE)
r = rotate(a, 15)
cv2.imwrite("../cv_segment/pics/rotate.jpg", r)
