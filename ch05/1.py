# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

# Logistic Classification

# Load data
xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([1, len(x_data)], -1.0, 1.0))

h = tf.matmul(W, X)
hypothesis = tf.div(1., 1. + tf.exp(-h))

# TODO
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(tf.Variable(0.1)).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in xrange(5001):
    fd = {
        X: x_data,
        Y: y_data,
    }
    sess.run(train, feed_dict=fd)
    if i % 20 == 0:
        print(i, sess.run(cost, feed_dict=fd), sess.run(W))

# Finally, ask to model!!
print('-' * 30)
# 2시간 공부하고 2번 수업에 들어간 친구는 낙제다.
print(sess.run(hypothesis, feed_dict={X: [[1], [2], [2]]}))
print(sess.run(hypothesis, feed_dict={X: [[1], [5], [5]]}))

# Try as matrix
print(sess.run(hypothesis, feed_dict={X: [[1, 1], [4, 3], [3, 5]]}))
