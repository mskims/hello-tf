import tensorflow as tf
from matplotlib import pyplot as plt



# Graph input

X = [1., 2., 3.]
Y = [1., 2., 3.]

m = n_samples = len(X)


W = tf.placeholder(tf.float32)

# y = WX
hypothesis = tf.mul(X, W)

# pow = AnB
cost = tf.reduce_mean(tf.pow(hypothesis-Y, 2))/(m)

init = tf.initialize_all_variables()

W_val = []
cost_val = []

sess = tf.Session()
sess.run(init)

for i in range(-30, 50):
    i *= 0.1
    print(i, sess.run(cost, feed_dict={W: i}))
    W_val.append(i)
    cost_val.append(sess.run(cost, feed_dict={W: i}))

# Graphic
plt.plot(W_val, cost_val, 'ro')
plt.ylabel('COST')
plt.xlabel('w')
plt.show()