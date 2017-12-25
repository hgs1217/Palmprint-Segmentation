# @Author:      HgS_1217_
# @Create Date: 2017/12/20

import tensorflow as tf
import numpy as np
import random
from config import CKPT_PATH


def conv2d(x, w, stride, padding='SAME'):
    return tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=padding)


def get_bilinear_filter(shape, factor):
    ksize = shape[1]
    center_loc = factor - 1 if ksize % 2 == 1 else factor - 0.5
    bilinear = np.zeros([shape[0], shape[1]])
    for x in range(shape[0]):
        for y in range(shape[1]):
            value = (1 - abs((x - center_loc) / factor)) * (1 - abs((y - center_loc) / factor))
            bilinear[x, y] = value

    weights = np.zeros(shape)
    for i in range(shape[2]):
        weights[:, :, i, i] = bilinear
    return tf.get_variable(name="bilinear_filter",
                           initializer=tf.constant_initializer(value=weights, dtype=tf.float32),
                           shape=weights.shape)


def upsample(x, width, channels, factor, name, padding="SAME"):
    ksize = 2 * factor - factor % 2
    stride = factor
    with tf.variable_scope(name) as scope:
        x_shape = x.get_shape()
        h, w = int(x_shape[1]) * stride, int(x_shape[2]) * stride
        weights = get_bilinear_filter([ksize, ksize, channels, channels], factor)
        return tf.nn.conv2d_transpose(x, weights, [width, h, w, channels], strides=[1, stride, stride, 1],
                                      padding=padding, name=scope.name)


def reshape_layer(x, num_classes, factor, width, name):
    with tf.variable_scope(name):
        w = tf.get_variable("w", [1, 1, int(x.get_shape()[3]), num_classes], dtype="float")
        b = tf.get_variable("b", [num_classes])
        resized = tf.nn.bias_add(conv2d(x, w, 1), b)

        return upsample(resized, width, num_classes, factor, name + "_upsampled")


def conv_layer(x, ksize, stride, feature_num, name, padding="SAME", groups=1):
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", [ksize, ksize, int(x.get_shape()[-1]) / groups, feature_num], dtype="float")
        b = tf.get_variable("b", [feature_num], dtype="float")

        x_split = tf.split(x, groups, 3)
        w_split = tf.split(w, groups, 3)

        feature_map_list = [conv2d(x_, w_, stride, padding) for x_, w_ in zip(x_split, w_split)]
        feature_map = tf.concat(feature_map_list, 3)

        out = tf.nn.bias_add(feature_map, b)
        feature_shape = list(map(lambda x: -1 if not x else x, feature_map.get_shape().as_list()))
        return tf.nn.relu(tf.reshape(out, feature_shape), name=scope.name)


def max_pool_layer(x, ksize, stride, name, padding="SAME"):
    return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1], padding=padding, name=name)


def fc_layer(x, in_dim, out_dim, relu_flag, name):
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", [in_dim, out_dim], dtype="float")
        b = tf.get_variable("b", [out_dim], dtype="float")
        out = tf.nn.xw_plus_b(x, w, b, name=scope.name)
        return tf.nn.relu(out) if relu_flag else out


def norm(x, lsize, bias=1.0, alpha=1e-4, beta=0.75):
    return tf.nn.lrn(x, lsize, bias=bias, alpha=alpha, beta=beta)


class VGG16:
    def __init__(self, raws, labels, test_raws, test_labels, input_size=256, keep_pb=0.5, num_classes=2,
                 batch_size=100, epoch_size=100, learning_rate=0.001):
        """
        :param raws: path list of raw images
        :param labels: path list of labels
        :param test_raws: path list of test images
        :param test_labels: path list of test labels
        :param keep_pb: keep probability of dropout
        :param num_classes: number of result classes
        """
        self.raws = raws
        self.labels = labels
        self.test_raws = test_raws
        self.test_labels = test_labels
        self.keep_pb = keep_pb
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.logits = None
        self.softmax = None
        self.classes = None

        self.x = tf.placeholder(tf.float32, shape=[None, self.input_size, self.input_size, 1], name="input_x")
        self.y = tf.placeholder(tf.float32, shape=[None, self.input_size, self.input_size, self.num_classes],
                                name="input_y")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.width = tf.placeholder(tf.int32, name="width")
        self.build_network()

    def build_network(self):
        x_resh = tf.reshape(self.x, [-1, self.input_size, self.input_size, 1])

        conv1_1 = conv_layer(x_resh, 3, 1, 64, "conv1_1")
        conv1_2 = conv_layer(conv1_1, 3, 1, 64, "conv1_2")
        pool1 = max_pool_layer(conv1_2, 2, 2, "pool1")

        conv2_1 = conv_layer(pool1, 3, 1, 128, "conv2_1")
        conv2_2 = conv_layer(conv2_1, 3, 1, 128, "conv2_2")
        pool2 = max_pool_layer(conv2_2, 2, 2, "pool2")

        conv3_1 = conv_layer(pool2, 3, 1, 256, "conv3_1")
        conv3_2 = conv_layer(conv3_1, 3, 1, 256, "conv3_2")
        conv3_3 = conv_layer(conv3_2, 3, 1, 256, "conv3_3")
        pool3 = max_pool_layer(conv3_3, 2, 2, "pool3")

        conv4_1 = conv_layer(pool3, 3, 1, 512, "conv4_1")
        conv4_2 = conv_layer(conv4_1, 3, 1, 512, "conv4_2")
        conv4_3 = conv_layer(conv4_2, 3, 1, 512, "conv4_3")
        pool4 = max_pool_layer(conv4_3, 2, 2, "pool4")

        conv5_1 = conv_layer(pool4, 3, 1, 512, "conv5_1")
        conv5_2 = conv_layer(conv5_1, 3, 1, 512, "conv5_2")
        conv5_3 = conv_layer(conv5_2, 3, 1, 512, "conv5_3")
        pool5 = max_pool_layer(conv5_3, 2, 2, "pool5")

        conv6 = conv_layer(pool5, 3, 1, 4096, "conv6")
        dropout6 = tf.nn.dropout(conv6, self.keep_prob)

        conv7 = conv_layer(dropout6, 1, 1, 4096, "conv7")
        dropout7 = tf.nn.dropout(conv7, self.keep_prob)

        layer7_reshape = reshape_layer(dropout7, self.num_classes, 32, self.width, "layer7_reshape")
        layer4_reshape = reshape_layer(pool4, self.num_classes, 16, self.width, "layer4_reshape")
        layer3_reshape = reshape_layer(pool3, self.num_classes, 8, self.width, "layer3_reshape")

        with tf.variable_scope("sum"):
            self.logits = tf.add(layer3_reshape, tf.add(2 * layer4_reshape, 4 * layer7_reshape))

        with tf.name_scope('result'):
            self.softmax = tf.nn.softmax(self.logits)
            self.classes = tf.argmax(self.softmax, axis=3)

    def load_img(self, path_list, label_flag):
        if label_flag:
            images = []
            for path in path_list:
                bi_img = tf.round(tf.image.convert_image_dtype(tf.image.decode_jpeg(
                    tf.read_file(path), channels=1), dtype=tf.uint8) / 255)
                images.append(tf.concat([1 - bi_img, bi_img], -1).eval())
            return images
        return [tf.image.convert_image_dtype(tf.image.decode_jpeg(
            tf.read_file(path), channels=1), dtype=tf.uint8).eval() for path in path_list]

    def batch_generator(self):
        rand_num = random.sample(range(len(self.raws)), self.batch_size)
        batch_raws, batch_labels = [self.raws[i] for i in rand_num], [self.labels[i] for i in rand_num]
        return self.load_img(batch_raws, False), self.load_img(batch_labels, True)

    def test_generator(self, i, total):
        isize = 10
        raws_test, labels_test = self.test_raws[i * isize: i * isize + isize], \
                                 self.test_labels[i * isize: i * isize + isize]
        return self.load_img(raws_test, False), self.load_img(labels_test, True)

    def get_optimizer(self):
        with tf.variable_scope("optimizer"):
            labels_reshape = tf.reshape(self.y, [-1, self.num_classes])
            logits_reshape = tf.reshape(self.logits, [-1, self.num_classes])
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels_reshape,
                                                             logits=logits_reshape)
            loss = tf.reduce_mean(losses)

            correct_prediction = tf.square(self.classes - tf.argmax(self.y, 3))
            accuracy = 1 - tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            optimizer = optimizer.minimize(loss)

        return optimizer, loss, accuracy

    def train(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            optimizer, loss, accuracy = self.get_optimizer()
            label_mapper = tf.argmax(self.y, axis=3)

            saver = tf.train.Saver()
            tf.add_to_collection('logits', self.logits)
            tf.add_to_collection('result', self.classes)
            sess.run(tf.global_variables_initializer())

            for i in range(self.epoch_size):
                batch_xs, batch_ys = self.batch_generator()
                loss_batch, accu, _ = sess.run([loss, accuracy, optimizer],
                                                feed_dict={self.x: batch_xs, self.y: batch_ys,
                                                           self.keep_prob: self.keep_pb, self.width: len(batch_ys)})
                print("train %d, loss %g, accu %g" % (i, loss_batch, accu))
                if i % 5 == 4:
                    total = 1
                    train_loss, train_accu = np.zeros(total), np.zeros(total)
                    for j in range(total):
                        x_test, y_test = self.test_generator(j, total)
                        loss_batch, accu = sess.run([loss, accuracy],
                                                        feed_dict={self.x: x_test, self.y: y_test,
                                                                   self.keep_prob: 1.0,
                                                                   self.width: len(y_test)})
                        train_loss[j], train_accu = loss_batch, accuracy
                    print("test: train %d, loss %g, accu %g" % (i, loss_batch, accu))

            saver.save(sess, CKPT_PATH)
            # total = 100
            # train_loss, train_accu = np.zeros(total), np.zeros(total)
            # for j in range(total):
            #     x_test, y_test = self.test_generator(j, total)
            #     loss_batch, accu = sess.run([loss, accuracy],
            #                                     feed_dict={self.x: x_test, self.y: y_test,
            #                                                self.keep_prob: 1.0,
            #                                                self.width: len(y_test)})
            #     train_loss[j], train_accu = loss_batch, accuracy
            # print("train %d, loss %g, accu %g" % (i, loss_batch, accu))
