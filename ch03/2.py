import tensorflow as tf
# Graph input


x_data = [1., 2., 3.]
y_data = [1., 2., 3.]

W = tf.Variable(tf.random_uniform([1], -10.0, 10.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# y = WX
# Why he use literal mul func instead of tf.mul ?
hypothesis = W * X

# Why example code uses square function instead of pow ?
cost = tf.reduce_mean(tf.square(hypothesis-Y))

# min
descent = W - tf.mul(0.1, tf.reduce_mean(tf.mul((tf.mul(W, X) - Y), X)))
# like define ?
update = W.assign(descent)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in xrange(50):
    fd = {
        X: x_data,
        Y: y_data
    }
    sess.run(update, feed_dict=fd)
    print(step, sess.run(cost, feed_dict=fd), sess.run(W))