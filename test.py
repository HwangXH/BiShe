import tensorflow as tf
from util_filters import lrelu, tanh_range, lerp
import cv2
import tensorflow as tf
import numpy as np

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def main():

    labels = np.array([0., 1., 0., 0.])
    logits = np.array([2., 3., 4., 1.])
    with tf.Session() as sess:
        print(sess.run(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)))

    p = sigmoid(logits)
    loss = -labels * np.log(p) - (1 - labels) * np.log(1 - p)
    print(loss)
    return 0


if __name__ == '__main__': main()