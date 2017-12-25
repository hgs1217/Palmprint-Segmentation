# @Author:      HgS_1217_
# @Create Date: 2017/12/21

import os
from config import OUTPUT_DIR, RESIZE_DIR, CON_OUTPUT_DIR, CON_RESIZE_DIR
from cnn.segnet import SegNet
import numpy as np


path = "D:/Computer Science/dataset/palmprint/output/0005_m_l_01.jpg"


def main():
    raws_total, labels_total = [], []
    for parent, dirnames, filenames in os.walk(CON_OUTPUT_DIR):
        labels_total = ["%s/%s" % (CON_OUTPUT_DIR, name)
                        for name in (filter(lambda x: x.split(".")[-1] == "jpg", filenames))]
    for parent, dirnames, filenames in os.walk(CON_RESIZE_DIR):
        raws_total = ["%s/%s" % (CON_RESIZE_DIR, name)
                      for name in (filter(lambda x: x.split(".")[-1] == "jpg", filenames))]

    raws, raws_test = raws_total[:-16], raws_total[-20:]
    labels, labels_test = labels_total[:-16], labels_total[-20:]
    # raws, raws_test = raws_total[16:32], raws_total[-16:]
    # labels, labels_test = labels_total[16:32], labels_total[-16:]

    segnet = SegNet(raws, labels, raws_test, labels_test, batch_size=20, epoch_size=50, input_size=128)
    segnet.train_network(True)
    # segnet.check()


if __name__ == '__main__':
    main()
