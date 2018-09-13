import numpy as np
import tensorflow as tf

x = [v for v in range(10)]


tx = tf.placeholder(tf.float32,shape = [10])
nx = tf.layers.batch_normalization(tx)
mean , variance = tf.nn.moments(tx,axes = [0])

normv = tf.nn.batch_normalization(tx,mean,variance,0,1,0)
with tf.Session() as sess:
    
  tf.global_variables_initializer().run()
  print(sess.run([mean,variance],feed_dict={tx:x}))
  print(sess.run(normv,feed_dict={tx:x}))


  value = sess.run(nx,feed_dict={tx:x});
  print(value)