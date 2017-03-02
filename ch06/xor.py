import tensorflow as tf
import numpy as np


xy_data = np.loadtxt('xor.txt', unpack=True, dtype='float32')
x_data = xy_data[:-1]
y_data = xy_data[-1]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
W = tf.Variable(tf.zeros([2], None))

h = tf.matmul(X, W)
hypothesis = tf.div(1., 1. + tf.exp(-h))
