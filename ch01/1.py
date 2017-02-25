import tensorflow as tf

hello = tf.constant('hello tensorflow!')


# START TF SESSION
sess = tf.Session()

print(sess.run(hello))