# @Author:      HgS_1217_
# @Create Date: 2017/12/21

import os
from config import OUTPUT_DIR, RESIZE_DIR, CON_OUTPUT_DIR, CON_RESIZE_DIR
from cnn.segnet import SegNet
import numpy as np


path = "D:/Computer Science/dataset/palmprint/output/0005_m_l_01.jpg"


def main():
    raws_total, labels_total = [], []
    for parent, dirnames, filenames in os.walk(OUTPUT_DIR):
        labels_total = ["%s/%s" % (OUTPUT_DIR, name)
                        for name in (filter(lambda x: x.split(".")[-1] == "jpg", filenames))]
    for parent, dirnames, filenames in os.walk(RESIZE_DIR):
        raws_total = ["%s/%s" % (RESIZE_DIR, name)
                      for name in (filter(lambda x: x.split(".")[-1] == "jpg", filenames))]

    total = list(range(64))
    raws_items = list(filter(lambda x: x % 8 != 6 and x % 8 != 7, total))
    test_items = list(set(total) ^ set(raws_items))

    raws, raws_test = [raws_total[i] for i in raws_items], [raws_total[i] for i in test_items]
    labels, labels_test = [labels_total[i] for i in raws_items], [labels_total[i] for i in test_items]
    # raws, raws_test = raws_total[:-16], raws_total[-16:]
    # labels, labels_test = labels_total[:-16], labels_total[-16:]

    # segnet = SegNet(raws, labels, raws_test, labels_test, batch_size=20, epoch_size=60, input_size=128)
    segnet = SegNet(raws, labels, raws_test, labels_test, batch_size=1, epoch_size=1000, input_size=128)
    segnet.train_network(True)
    # segnet.check()


if __name__ == '__main__':
    main()
