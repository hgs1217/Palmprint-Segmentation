# @Author:      HgS_1217_
# @Create Date: 2017/12/21

import os
from config import OUTPUT_DIR, RESIZE_DIR, CON_OUTPUT_DIR, CON_RESIZE_DIR
from cnn.segnet import SegNet
import cv2


def main():
    raws_total, labels_total = [], []
    for parent, dirnames, filenames in os.walk(OUTPUT_DIR):
        labels_total = ["%s/%s" % (OUTPUT_DIR, name)
                        for name in (filter(lambda x: x.split(".")[-1] == "jpg", filenames))]
    for parent, dirnames, filenames in os.walk(RESIZE_DIR):
        raws_total = ["%s/%s" % (RESIZE_DIR, name)
                      for name in (filter(lambda x: x.split(".")[-1] == "jpg", filenames))]

    total = list(range(len(raws_total)))
    raws_items = list(filter(lambda x: x % 5 != 4, total))
    test_items = list(set(total) ^ set(raws_items))

    raws, raws_test = [raws_total[i] for i in raws_items], [raws_total[i] for i in test_items]
    labels, labels_test = [labels_total[i] for i in raws_items], [labels_total[i] for i in test_items]

    segnet = SegNet(raws, labels, raws_test, labels_test, start_step=0, batch_size=1, epoch_size=600, input_size=128)
    segnet.train_network(True)
    # img = cv2.imread("D:/Computer Science/Github/Palmprint-Segmentation/cv_segment/pics/canny1.jpg", cv2.IMREAD_GRAYSCALE)
    # out = segnet.check(img)
    # cv2.imwrite("D:/Computer Science/Github/Palmprint-Segmentation/cv_segment/pics/net_res.jpg", out)


if __name__ == '__main__':
    main()
