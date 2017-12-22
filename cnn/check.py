# @Author:      HgS_1217_
# @Create Date: 2017/12/22

import tensorflow as tf
import numpy as np
import random
import cv2

FCN_PATH = 'D:/Computer Science/dataset/palmprint/fcn-vgg/'
IMG_PATH = "D:/Computer Science/dataset/palmprint/test_raw/0018_m_r_08.jpg"


def main(path):
    ckpt = tf.train.get_checkpoint_state(path)
    saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path +'.meta')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    graph = tf.get_default_graph()

    x = graph.get_operation_by_name('input_x').outputs[0]
    y = tf.get_collection('result')[0]
    keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]
    width = graph.get_operation_by_name('width').outputs[0]

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt.model_checkpoint_path)

        img_gray = sess.run(tf.image.convert_image_dtype(tf.image.decode_jpeg(
            tf.read_file(IMG_PATH), channels=1), dtype=tf.uint8))
        img_np = tf.reshape(img_gray, [-1, 256, 256, 1]).eval()

        result = sess.run(y, feed_dict={x: img_np, keep_prob: 1.0, width: 1})
        out = np.array(result[0]) * 255

    print(out[120:140, 120:140])
    cv2.imwrite("D:/Computer Science/Github/Palmprint-Segmentation/cv_segment/pics/net_res.jpg", out)

if __name__ == '__main__':
    main(FCN_PATH)
