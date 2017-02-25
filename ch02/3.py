# -*- coding: utf-8 -*-

import tensorflow as tf

x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = W * X + b

# (hypothesis - REAL_DATA)n2
# 예상치와 실제 데이터값의 차이를 계산하는 라인.
cost = tf.reduce_mean(tf.square(hypothesis - Y))

a = tf.Variable(0.1)
# 그라디언트를 따라 따라 내려가는 옵티마이저 함수 (학습을 최적화하는 함수)
optimizer = tf.train.GradientDescentOptimizer(a)

# 기울기와 b값을 최소화 하는 코스트함수
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W), sess.run(b))


# Results
print(sess.run(hypothesis, feed_dict={X: 6}))
print(sess.run(hypothesis, feed_dict={X: 12}))
