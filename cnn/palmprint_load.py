# @Author:      HgS_1217_
# @Create Date: 2017/12/21

import os
from config import OUTPUT_DIR, RESIZE_DIR
from cnn.vgg16 import VGG16
import numpy as np


path = "D:/Computer Science/dataset/palmprint/output/0002_m_l_01.jpg"


def main():
    raws_total, labels_total = [], []
    for parent, dirnames, filenames in os.walk(OUTPUT_DIR):
        labels_total = ["%s/%s" % (OUTPUT_DIR, name)
                        for name in (filter(lambda x: x.split(".")[-1] == "jpg", filenames))]
    for parent, dirnames, filenames in os.walk(RESIZE_DIR):
        raws_total = ["%s/%s" % (RESIZE_DIR, name)
                      for name in (filter(lambda x: x.split(".")[-1] == "jpg", filenames))]

    # raws, raws_test = raws_total[:-1000], raws_total[-1000:]
    # labels, labels_test = labels_total[:-1000], labels_total[-1000:]
    raws, raws_test = raws_total[:-16], raws_total[-16:]
    labels, labels_test = labels_total[:-16], labels_total[-16:]

    vgg16 = VGG16(raws, labels, raws_test, labels_test, batch_size=7, epoch_size=200)
    vgg16.train()


if __name__ == '__main__':
    main()
    # import tensorflow as tf
    #
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    #
    # with tf.Session(config=config) as sess:
    #     a = tf.round(tf.image.convert_image_dtype(tf.image.decode_jpeg(
    #         tf.read_file(path), channels=1), dtype=tf.uint8) / 255)
    #     b = tf.concat([1 - a, a], -1)
    #
    #     x = tf.zeros([256, 256, 1])
    #     y = tf.concat([1 - x, x], -1)
    #
    #     c = tf.argmax(b, 2)
    #     z = tf.argmax(y, 2)
    #
    #     g = 1 - tf.reduce_sum(tf.square(c - z)) / 256 / 256
    #
    #     print(sess.run(c[110:130, 110:130]))
    #     print(sess.run(z[110:130, 110:130]))
    #     print(sess.run(g))

