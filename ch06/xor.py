import tensorflow as tf
import numpy as np

xy_data = np.loadtxt('xor.txt', unpack=True)
x_data = xy_data[0:-1]
y_data = xy_data[-1]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
W = tf.Variable(tf.random_uniform([1, len(x_data)], -1.0, 1.0))

h = tf.matmul(W, X)
# Sigmoid Function
hypothesis = tf.div(1., 1 + tf.exp(-h))

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(tf.Variable(0.1)).minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in xrange(2001):
    fd = {
        X: x_data,
        Y: y_data,
    }
    sess.run(train, feed_dict=fd)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(cost, feed_dict=fd))

