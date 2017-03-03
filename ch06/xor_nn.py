import tensorflow as tf
import numpy as np


# XOR EXAM
"""
    2017, Mar, 3 Fri
    Accuracy is can not be close at 1.0.
    we can now know limit of Logistic regression.
"""

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

fd = {
    X: x_data,
    Y: y_data,
}
for step in xrange(2001):
    sess.run(train, feed_dict=fd)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(cost, feed_dict=fd))

# Test model
correct_prediction = tf.equal(tf.floor(hypothesis + 0.5), Y)

# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

print(sess.run([hypothesis, tf.floor(hypothesis + 0.5), correct_prediction, accuracy], feed_dict=fd))
print("Accuracy : ", sess.run(accuracy, feed_dict=fd))
