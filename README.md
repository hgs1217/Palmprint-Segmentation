# Palmprint-Segmentation

Palmprint Segmentation based on Bayesian SegNet

Github:  https://github.com/hgs1217/Palmprint-Segmentation

## Requirements

- Python 3.+
- Numpy
- Opencv for Python3
- Tensorflow-gpu

## Usage

To run the project, "main.py" is the entrance, and you have to set your input image path and the output image path.
As tensorflow model is required for running, you have to set the model ckpt file path in "config.py".

If you want to train your own segmentation model, you can run "cnn/palmprint_load.py" to start the training.

The dataset of our training is based on CASIA Palmprint Database, you can download at http://www.cbsr.ia.ac.cn/china/Palmprint\%20Databases\%20CH.asp
