# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

xy_data = np.loadtxt('train.txt', unpack=True, dtype='float32')
x_data = np.transpose(xy_data[0:3])
y_data = np.transpose(xy_data[3:])

X = tf.placeholder("float", [None, 3])
Y = tf.placeholder("float", [None, 3])

W = tf.Variable(tf.zeros([3, 3]))

# Model
hypothesis = tf.nn.softmax(tf.matmul(X, W))

# cost = tf.reduce_mean
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), reduction_indices=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    fd = {
        X: x_data,
        Y: y_data,
    }
    sess.run(init)
    for step in xrange(100001):
        sess.run(optimizer, feed_dict=fd)
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict=fd), sess.run(W))

    print('-*'*30)
    # TEST and one-hot encoding
    a = sess.run(hypothesis, feed_dict={X: [[1, 10, 11]]})
    print(a, sess.run(tf.arg_max(a, 1)))

    b = sess.run(hypothesis, feed_dict={X: [[1, 10, 2]]})
    print(b, sess.run(tf.arg_max(b, 1)))

    c = sess.run(hypothesis, feed_dict={X: [[1, 1, 3]]})
    print(c, sess.run(tf.arg_max(c, 1)))

    all = sess.run(hypothesis, feed_dict={X: [[1, 10, 11], [1, 4, 50], [1, 1, 3]]})
    print(all, sess.run(tf.arg_max(all, 1)))
