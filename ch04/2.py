import tensorflow as tf

# single-variable
# H(x) = Wx + b

# Multi-variable-feature
# H(x1, x2) = w1x1 + w2x2 + b

# Those functions use same function => so, the costs are also same

# hypothesis........ will be long
# So we decided to use 'Matrix' (hang-ryul)


# gkrtmq data
x_data = [
    # b in Matrix
    [1., 1., 1., 1., 1.],
    [1., 0., 3., 0., 5.],
    [0., 2., 0., 4., 0.]
]
y_data = [1, 2, 3, 4, 5]

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

# Launch grapth
sess = tf.Session()
sess.run(init)

for step in xrange(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W))
