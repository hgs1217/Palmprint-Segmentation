# @Author:      HgS_1217_
# @Create Date: 2017/12/23

import numpy as np
import tensorflow as tf


def variable_with_weight_decay(name, shape, initializer, wd):
    var = tf.get_variable(name, shape, initializer=initializer)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def get_hist(predictions, labels):
    num_class, batch_size = predictions.shape[3], predictions.shape[0]
    hist = np.zeros((num_class, num_class))
    for i in range(batch_size):
        hist += fast_hist(labels[i].flatten(), predictions[i].argmax(2).flatten(), num_class)
    return hist


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_acc(predictions, label_tensor):
    labels = label_tensor
    size, num_class = predictions.shape[0], predictions.shape[3]
    hist = np.zeros((num_class, num_class))
    for i in range(size):
        hist += fast_hist(labels[i].flatten(), predictions[i].argmax(2).flatten(), num_class)
    acc_total = np.diag(hist).sum() / hist.sum()
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print('\taccuracy = %f\t\tmean IU  = %f' % (np.nanmean(acc_total), np.nanmean(iu)))
    for ii in range(num_class):
        if float(hist.sum(1)[ii]) == 0:
            acc = 0.0
        else:
            acc = np.diag(hist)[ii] / float(hist.sum(1)[ii])
        print("\tclass # %d accuracy = %f " % (ii, acc), end="")
    print("")


def print_hist_summary(hist):
    acc_total = np.diag(hist).sum() / hist.sum()
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print('\taccuracy = %f\t\tmean IU  = %f' % (np.nanmean(acc_total), np.nanmean(iu)))
    for ii in range(hist.shape[0]):
        if float(hist.sum(1)[ii]) == 0:
            acc = 0.0
        else:
            acc = np.diag(hist)[ii] / float(hist.sum(1)[ii])
        print("\tclass # %d accuracy = %f " % (ii, acc), end="")
    print("")
