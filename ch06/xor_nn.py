import tensorflow as tf
import numpy as np

xy_data = np.loadtxt('xor.txt', unpack=True)
x_data = np.transpose(xy_data[0:-1])
y_data = np.reshape(xy_data[-1], (4, 1))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

b1 = tf.zeros([2], name='Bias1')
b2 = tf.zeros([1], name='Bias2')

W1 = tf.Variable(tf.random_uniform([2, 2], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([2, 1], -1.0, 1.0))

L2 = tf.sigmoid( tf.matmul(X, W1) + b1)
hypothesis = tf.sigmoid( tf.matmul(L2, W2) + b2)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(tf.Variable(0.1)).minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

fd = {
    X: x_data,
    Y: y_data,
}
for step in xrange(100000):
    sess.run(train, feed_dict=fd)
    if step % 200 == 0:
        print(step, sess.run([W1, W2]), sess.run(cost, feed_dict=fd))

# Test model
correct_prediction = tf.equal(tf.floor(hypothesis + 0.5), Y)

# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

print(sess.run([hypothesis, tf.floor(hypothesis + 0.5), correct_prediction, accuracy], feed_dict=fd))
print("Accuracy : ", sess.run(accuracy, feed_dict=fd))
