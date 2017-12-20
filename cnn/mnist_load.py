# @Author:      HgS_1217_
# @Create Date: 2017/12/20

import os
import struct
import numpy as np
from cnn.alexnet import Alexnet


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784).astype(np.float32)

    labels = [[0] * i + [1] + [0] * (9 - i) for i in labels]

    return (images - images.min()) / (images.max() - images.min()), labels


def main():
    images, labels = load_mnist("D:/Computer Science/Github/Mnist-tensorflow/")
    images_test, labels_test = load_mnist("D:/Computer Science/Github/Mnist-tensorflow/", "t10k")

    alexnet = Alexnet(images, labels, images_test, labels_test, 0.5, 100, 300)
    alexnet.train()


if __name__ == '__main__':
    main()
