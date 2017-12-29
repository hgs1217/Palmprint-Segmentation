# @Author:      HgS_1217_
# @Create Date: 2017/12/23

# @Author:      HgS_1217_
# @Create Date: 2017/12/20

import tensorflow as tf
import numpy as np
import random
import math
import time
import cv2
from cnn.network_utils import variable_with_weight_decay, add_loss_summaries, per_class_acc, get_hist, \
    print_hist_summary
from config import CKPT_PATH, LOG_PATH


def msra_initializer(ksize, filter_num):
    stddev = math.sqrt(2. / (ksize ** 2 * filter_num))
    return tf.truncated_normal_initializer(stddev=stddev)


def orthogonal_initializer(scale=1.1):
    """
    From Lasagne and Keras. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    """
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)

    return _initializer


def get_deconv_filter(shape):
    width, height = shape[0], shape[1]
    f = math.ceil(width / 2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([shape[0], shape[1]])
    for x in range(width):
        for y in range(height):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(shape)
    for i in range(shape[2]):
        weights[:, :, i, i] = bilinear

    return tf.get_variable(name="up_filter", initializer=tf.constant_initializer(value=weights, dtype=tf.float32),
                           shape=weights.shape)


def weighted_loss(lgts, lbs, num_classes):
    with tf.name_scope('loss'):
        logits = tf.reshape(lgts, (-1, num_classes))
        labels = tf.cast(lbs, tf.int32)
        epsilon = tf.constant(value=1e-10)

        logits = logits + epsilon
        label_flat = tf.reshape(labels, (-1, 1))
        labels = tf.reshape(tf.one_hot(label_flat, depth=num_classes), (-1, num_classes))

        cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(tf.nn.softmax(logits) + epsilon),
                                                   np.array([0.625, 2.5])), axis=[1])
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

        tf.add_to_collection('losses', cross_entropy_mean)
        return tf.add_n(tf.get_collection('losses'), name='total_loss')


def conv2d(x, w, stride, padding='SAME'):
    return tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=padding)


def batch_norm_layer(x, is_training, scope):
    return tf.cond(is_training,
                   lambda: tf.contrib.layers.batch_norm(x, is_training=True, center=False,
                                                        updates_collections=None, scope=scope + "_bn"),
                   lambda: tf.contrib.layers.batch_norm(x, is_training=False, updates_collections=None,
                                                        center=False, scope=scope + "_bn", reuse=True))


def deconv_layer(x, ksize, channels, output_shape, stride=2, name=None):
    with tf.variable_scope(name):
        weights = get_deconv_filter([ksize, ksize, channels, channels])
        return tf.nn.conv2d_transpose(x, weights, output_shape, strides=[1, stride, stride, 1],
                                      padding='SAME')


def conv_layer(x, ksize, stride, feature_num, is_training, name=None, padding="SAME", relu_flag=True,
               in_channel=None):
    channel = int(x.get_shape()[-1]) if not in_channel else in_channel
    with tf.variable_scope(name) as scope:
        w = variable_with_weight_decay('w', shape=[ksize, ksize, channel, feature_num],
                                       initializer=orthogonal_initializer(), wd=None)
        b = tf.get_variable("b", [feature_num], initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv2d(x, w, stride, padding), b)
        norm = batch_norm_layer(bias, is_training, scope.name)
        return tf.nn.relu(norm) if relu_flag else norm


def max_pool_layer(x, ksize, stride, name, padding="SAME"):
    return tf.nn.max_pool_with_argmax(x, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1],
                                      padding=padding, name=name)


def norm_layer(x, lsize, bias=1.0, alpha=1e-4, beta=0.75, name=None):
    return tf.nn.lrn(x, lsize, bias=bias, alpha=alpha, beta=beta, name=name)


class SegNet:
    def __init__(self, raws, labels, test_raws, test_labels, input_size=256, keep_pb=0.5, num_classes=2,
                 batch_size=1, epoch_size=100, learning_rate=0.001, start_step=0):
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
        self.loss = None
        self.start_step = start_step

        self.x = tf.placeholder(tf.float32, shape=[None, self.input_size, self.input_size, 1],
                                name="input_x")
        self.y = tf.placeholder(tf.float32, shape=[None, self.input_size, self.input_size, 1],
                                name="input_y")
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        self.width = tf.placeholder(tf.int32, name="width")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.global_step = tf.Variable(0, trainable=False)

    def build_network(self, images, labels, batch_size, is_training):
        norm1 = norm_layer(images, 5, name="norm1")
        conv1_1 = conv_layer(norm1, 3, 1, 64, is_training, "conv1_1")
        conv1_2 = conv_layer(conv1_1, 3, 1, 64, is_training, "conv1_2")
        pool1, pool1_indices = max_pool_layer(conv1_2, 2, 2, "pool1")

        conv2_1 = conv_layer(pool1, 3, 1, 128, is_training, "conv2_1")
        conv2_2 = conv_layer(conv2_1, 3, 1, 128, is_training, "conv2_2")
        pool2, pool2_indices = max_pool_layer(conv2_2, 2, 2, "pool2")

        conv3_1 = conv_layer(pool2, 3, 1, 256, is_training, "conv3_1")
        conv3_2 = conv_layer(conv3_1, 3, 1, 256, is_training, "conv3_2")
        conv3_3 = conv_layer(conv3_2, 3, 1, 256, is_training, "conv3_3")
        pool3, pool3_indices = max_pool_layer(conv3_3, 2, 2, "pool3")
        encdrop3 = tf.nn.dropout(pool3, self.keep_prob, name="encdrop3")

        conv4_1 = conv_layer(encdrop3, 3, 1, 512, is_training, "conv4_1")
        conv4_2 = conv_layer(conv4_1, 3, 1, 512, is_training, "conv4_2")
        conv4_3 = conv_layer(conv4_2, 3, 1, 512, is_training, "conv4_3")
        pool4, pool4_indices = max_pool_layer(conv4_3, 2, 2, "pool4")
        encdrop4 = tf.nn.dropout(pool4, self.keep_prob, name="encdrop4")

        # conv5_1 = conv_layer(encdrop4, 3, 1, 512, is_training, "conv5_1")
        # conv5_2 = conv_layer(conv5_1, 3, 1, 512, is_training, "conv5_2")
        # conv5_3 = conv_layer(conv5_2, 3, 1, 512, is_training, "conv5_3")
        # pool5, pool5_indices = max_pool_layer(conv5_3, 2, 2, "pool5")
        # encdrop5 = tf.nn.dropout(pool5, self.keep_prob, name="encdrop5")
        #
        # upsample5 = deconv_layer(encdrop5, 2, 512, [batch_size, 8, 8, 512], name="upsample5")
        # conv_decode5_3 = conv_layer(upsample5, 3, 1, 512, is_training, "conv_decode5_3", relu_flag=False,
        #                           in_channel=512)
        # conv_decode5_2 = conv_layer(conv_decode5_3, 3, 1, 512, is_training, "conv_decode5_2", relu_flag=False,
        #                             in_channel=512)
        # conv_decode5_1 = conv_layer(conv_decode5_2, 3, 1, 512, is_training, "conv_decode5_1", relu_flag=False,
        #                             in_channel=512)
        # decdrop5 = tf.nn.dropout(conv_decode5_1, self.keep_prob, name="decdrop5")

        upsample4 = deconv_layer(encdrop4, 2, 512, [batch_size, 16, 16, 512], name="upsample4")
        conv_decode4_3 = conv_layer(upsample4, 3, 1, 512, is_training, "conv_decode4_3", relu_flag=False,
                                  in_channel=512)
        conv_decode4_2 = conv_layer(conv_decode4_3, 3, 1, 512, is_training, "conv_decode4_2", relu_flag=False,
                                    in_channel=512)
        conv_decode4_1 = conv_layer(conv_decode4_2, 3, 1, 256, is_training, "conv_decode4_1", relu_flag=False,
                                    in_channel=512)
        decdrop4 = tf.nn.dropout(conv_decode4_1, self.keep_prob, name="decdrop4")

        upsample3 = deconv_layer(decdrop4, 2, 256, [batch_size, 32, 32, 256], name="upsample3")
        conv_decode3_3 = conv_layer(upsample3, 3, 1, 256, is_training, "conv_decode3_3", relu_flag=False,
                                  in_channel=256)
        conv_decode3_2 = conv_layer(conv_decode3_3, 3, 1, 256, is_training, "conv_decode3_2", relu_flag=False,
                                    in_channel=256)
        conv_decode3_1 = conv_layer(conv_decode3_2, 3, 1, 128, is_training, "conv_decode3_1", relu_flag=False,
                                    in_channel=256)
        decdrop3 = tf.nn.dropout(conv_decode3_1, self.keep_prob, name="decdrop3")

        upsample2 = deconv_layer(decdrop3, 2, 128, [batch_size, 64, 64, 128], name="upsample2")
        conv_decode2_2 = conv_layer(upsample2, 3, 1, 128, is_training, "conv_decode2_2", relu_flag=False,
                                  in_channel=128)
        conv_decode2_1 = conv_layer(conv_decode2_2, 3, 1, 64, is_training, "conv_decode2_1", relu_flag=False,
                                    in_channel=128)

        upsample1 = deconv_layer(conv_decode2_1, 2, 64, [batch_size, 128, 128, 64], name="upsample1")
        conv_decode1_2 = conv_layer(upsample1, 3, 1, 64, is_training, "conv_decode1_2", relu_flag=False,
                                  in_channel=64)
        conv_decode1_1 = conv_layer(conv_decode1_2, 3, 1, 64, is_training, "conv_decode1_1", relu_flag=False,
                                    in_channel=64)

        with tf.variable_scope('conv_classifier') as scope:
            w = variable_with_weight_decay('w', shape=[1, 1, 64, self.num_classes],
                                           initializer=msra_initializer(1, 64), wd=0.0005)
            b = tf.get_variable("b", [self.num_classes], initializer=tf.constant_initializer(0.0))
            conv_classifier = tf.nn.bias_add(conv2d(conv_decode1_1, w, 1), b, name=scope.name)

        logits = conv_classifier
        loss = weighted_loss(conv_classifier, labels, self.num_classes)
        classes = tf.argmax(logits, axis=-1)

        return loss, logits, classes

    def train_set(self, total_loss, global_step):
        loss_averages_op = add_loss_summaries(total_loss)

        with tf.control_dependencies([loss_averages_op]):
            opt = tf.train.AdamOptimizer(self.learning_rate)
            grads = opt.compute_gradients(total_loss)

        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

        variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            train_op = tf.no_op(name='train')

        return train_op

    def load_img(self, path_list, label_flag):
        if label_flag:
            return tf.stack([np.round(tf.image.convert_image_dtype(tf.image.decode_jpeg(
                tf.read_file(path), channels=1), dtype=tf.uint8).eval() / 255) for path in path_list]).eval()
        return tf.stack([tf.image.convert_image_dtype(tf.image.decode_jpeg(
            tf.read_file(path), channels=1), dtype=tf.uint8).eval() for path in path_list]).eval()

    def batch_generator(self):
        rand_num = random.sample(range(len(self.raws)), self.batch_size)
        batch_raws, batch_labels = [self.raws[i] for i in rand_num], [self.labels[i] for i in rand_num]
        return self.load_img(batch_raws, False), self.load_img(batch_labels, True)

    def test_generator(self, i):
        isize = self.batch_size
        raws_test = self.test_raws[i * isize: i * isize + isize]
        labels_test = self.test_labels[i * isize: i * isize + isize]
        return self.load_img(raws_test, False), self.load_img(labels_test, True)

    def check_generator(self):
        batch_raws, batch_labels = [self.raws[i] for i in range(self.batch_size)], \
                                   [self.labels[i] for i in range(self.batch_size)]
        # rand_num = [-1]
        # batch_raws, batch_labels = [self.raws[i] for i in rand_num], [self.labels[i] for i in rand_num]
        return self.load_img(batch_raws, False), self.load_img(batch_labels, True)

    def train_network(self, is_finetune=False):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            loss, eval_prediction, classes = self.build_network(self.x, self.y, self.width, self.is_training)
            train_op = self.train_set(loss, self.global_step)

            tf.add_to_collection('result', classes)
            saver = tf.train.Saver(tf.global_variables())

            summary_op = tf.summary.merge_all()

            if (is_finetune):
                saver.restore(sess, CKPT_PATH)
            else:
                sess.run(tf.global_variables_initializer())

            summary_writer = tf.summary.FileWriter(LOG_PATH, sess.graph)
            average_pl = tf.placeholder(tf.float32)
            acc_pl = tf.placeholder(tf.float32)
            iu_pl = tf.placeholder(tf.float32)
            average_summary = tf.summary.scalar("test_average_loss", average_pl)
            acc_summary = tf.summary.scalar("test_accuracy", acc_pl)
            iu_summary = tf.summary.scalar("Mean_IU", iu_pl)

            min_loss = 9999
            loss_iter = 0
            vali_iter = 20
            for step in range(self.start_step, self.start_step + self.epoch_size):
                batch_xs, batch_ys = self.batch_generator()
                feed_dict = {self.x: batch_xs, self.y: batch_ys, self.width: self.batch_size,
                             self.is_training: True, self.keep_prob: self.keep_pb}

                start_time = time.time()
                loss_batch, eval_pre, _ = sess.run([loss, eval_prediction, train_op], feed_dict=feed_dict)
                duration = time.time() - start_time
                print("train %d, loss %g, duration %.3f" % (step, loss_batch, duration))
                per_class_acc(eval_pre, batch_ys)
                loss_iter += loss_batch / vali_iter

                if step % vali_iter == vali_iter - 1:
                    print("\nstart validating.....")
                    total_val_loss = 0.0
                    hist = np.zeros((self.num_classes, self.num_classes))
                    test_iter = 8
                    for test_step in range(test_iter):
                        x_test, y_test = self.test_generator(test_step)
                        loss_test, eval_pre = sess.run([loss, eval_prediction], feed_dict={
                            self.x: x_test,
                            self.y: y_test,
                            self.width: self.batch_size,
                            self.is_training: True,
                            self.keep_prob: 1.0
                        })
                        total_val_loss += loss_test
                        hist += get_hist(eval_pre, y_test)
                    print("val loss: ", total_val_loss / test_iter)
                    acc_total = np.diag(hist).sum() / hist.sum()
                    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
                    test_summary_str = sess.run(average_summary, feed_dict={average_pl: total_val_loss / test_iter})
                    acc_summary_str = sess.run(acc_summary, feed_dict={acc_pl: acc_total})
                    iu_summary_str = sess.run(iu_summary, feed_dict={iu_pl: np.nanmean(iu)})
                    print_hist_summary(hist)
                    print("end validating....\n")

                    summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.add_summary(test_summary_str, step)
                    summary_writer.add_summary(acc_summary_str, step)
                    summary_writer.add_summary(iu_summary_str, step)

                    print("last %d average loss: %g\n" % (vali_iter, loss_iter))
                    if loss_iter < min_loss:
                        min_loss = loss_iter
                        print("saving model.....")
                        saver.save(sess, CKPT_PATH)
                        print("end saving....\n")
                    loss_iter = 0

            print("saving model.....")
            saver.save(sess, CKPT_PATH)
            print("end saving....\n")

    def check(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            loss, eval_prediction, classes = self.build_network(self.x, self.y, self.width, self.is_training)

            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess, CKPT_PATH)

            batch_xs, batch_ys = self.check_generator()
            feed_dict = {self.x: batch_xs, self.y: batch_ys, self.width: 1, self.is_training: True,
                         self.keep_prob: 1.0}

            loss_batch, eval_pre, res = sess.run([loss, eval_prediction, classes], feed_dict=feed_dict)
            per_class_acc(eval_pre, batch_ys)

            # batch_ys = sess.run(tf.argmax(batch_ys, axis=-1))

        print(batch_ys[0].shape)
        c = np.zeros((128, 128), dtype=np.uint8)
        for i in range(128):
            for j in range(128):
                c[i, j] = batch_ys[0][i][j]

        print(c[0:20, 0:20])
        print(res[0].shape)
        print(res[0][0:20, 0:20])
        out = np.array(res[0]) * 255
        cv2.imwrite("D:/Computer Science/Github/Palmprint-Segmentation/cv_segment/pics/net_res2.jpg", out)
