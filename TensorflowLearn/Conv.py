import tensorflow as tf
layers = tf.contrib.layers

x1 = tf.constant(1.0, shape=[1,3,3,1])
kernel = tf.constant(1.0, shape=[3,3,3,1])
x2 = tf.constant(1.0, shape=[1,6,6,3])
x3 = tf.constant(1.0, shape=[1,5,5,3])

#y2 = tf.nn.conv2d(x3, kernel, strides=[1,2,2,1], padding="SAME")
y2 = tf.layers.conv2d(x3, 1, [4,4], strides=[2,2], padding="SAME")


a = tf.constant(1.0, shape=[1,28,28,32])
y3 = layers.conv2d(a, 1, 4)

noise = tf.random_normal([1,64])

net = layers.fully_connected(noise, 1024)
net = layers.fully_connected(net, 7 * 7 * 256)
net = tf.reshape(net, [-1, 7, 7, 256])
net = layers.conv2d_transpose(net, 64, [4, 4], stride=2)
net = layers.conv2d_transpose(net, 32, [4, 4], stride=2)
# Make sure that generator output is in the same range as `inputs`
# ie [-1, 1].
net = layers.conv2d(net, 1, 4, normalizer_fn=None, activation_fn=tf.tanh)

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(net))
