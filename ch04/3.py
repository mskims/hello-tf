import tensorflow as tf
import numpy as np


xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]

# SET VARIABLES
W = tf.Variable(tf.random_uniform([1, 3], -1.0, 1.0))

# Our hypothesis
# Use matmul to mul Matrix (W)
hypothesis = tf.matmul(W, x_data)

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(0.1)

train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

# Launch graph
sess = tf.Session()
sess.run(init)

for step in xrange(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W))
