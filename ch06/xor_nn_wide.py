import tensorflow as tf
import numpy as np

xy = np.loadtxt('xor.txt', unpack=True)

"""
array([[ 0.,  0.,  1.,  1.],
       [ 0.,  1.,  0.,  1.]])

TO

array([[ 0.,  0.],
       [ 0.,  1.],
       [ 1.,  0.],
       [ 1.,  1.]])
"""
x_data = np.transpose(xy[0:-1])

"""
array([ 0.,  1.,  1.,  0.])

TO

array([[ 0.],
       [ 1.],
       [ 1.],
       [ 0.]])
"""
y_data = np.reshape(xy[-1], (4, 1))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([2, 10], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([10, 1], -1.0, 1.0))

b1 = tf.Variable(tf.zeros([10]))
b2 = tf.Variable(tf.zeros([1]))

L2 = tf.sigmoid(tf.matmul(X, W1) + b1)
hypothesis = tf.sigmoid(tf.matmul(L2, W2) + b2)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(tf.Variable(0.1)).minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

fd = {
    X: x_data,
    Y: y_data,
}
for step in xrange(4000):
    sess.run(train, feed_dict=fd)
    if step % 200 == 0:
        print(step, sess.run(cost, feed_dict=fd), sess.run(W1), sess.run(W2))

print('=' * 50)
print('Calculate Accuracy')
print('=' * 50)

correct_prediction = tf.equal(tf.floor(hypothesis + 0.5), Y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
print(sess.run([hypothesis, accuracy], feed_dict=fd), sess.run([W1, W2]))
