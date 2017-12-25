# @Author:      HgS_1217_
# @Create Date: 2017/12/22

import tensorflow as tf
import numpy as np
import random
import cv2
import os
from config import OUTPUT_DIR, RESIZE_DIR

NET_PATH = 'D:/Computer Science/dataset/palmprint/'
IMG_PATH = "D:/Computer Science/dataset/palmprint/small_raw/0005_m_l_02.jpg"


def main(path):
    raws_total, labels_total = [], []
    for parent, dirnames, filenames in os.walk(OUTPUT_DIR):
        labels_total = ["%s/%s" % (OUTPUT_DIR, name)
                        for name in (filter(lambda x: x.split(".")[-1] == "jpg", filenames))]
    for parent, dirnames, filenames in os.walk(RESIZE_DIR):
        raws_total = ["%s/%s" % (RESIZE_DIR, name)
                      for name in (filter(lambda x: x.split(".")[-1] == "jpg", filenames))]

    raws = raws_total[16]
    labels = labels_total[16:32]

    ckpt = tf.train.get_checkpoint_state(path)
    saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path +'.meta')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    graph = tf.get_default_graph()

    x = graph.get_operation_by_name('input_x').outputs[0]
    y = tf.get_collection('result')[0]
    width = graph.get_operation_by_name('width').outputs[0]
    is_training = graph.get_operation_by_name('is_training').outputs[0]

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt.model_checkpoint_path)

        img_gray = sess.run(tf.image.convert_image_dtype(tf.image.decode_jpeg(
            tf.read_file(IMG_PATH), channels=1), dtype=tf.uint8))
        img = tf.reshape(img_gray, [-1, 128, 128, 1]).eval()

        result = sess.run(y, feed_dict={x: img, is_training: True,
                                        width: 1})
        print(result[0])
        out = np.array(result[0]) * 255

    cv2.imwrite("D:/Computer Science/Github/Palmprint-Segmentation/cv_segment/pics/net_res3.jpg", out)

if __name__ == '__main__':
    main(NET_PATH)
