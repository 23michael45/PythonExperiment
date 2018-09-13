import tensorflow as tf 
import numpy as np

x = tf.Variable(0.0)
#返回一个op，表示给变量x加1的操作
x_plus_1 = tf.assign_add(x, 1)
  
#control_dependencies的意义是，在执行with包含的内容（在这里就是 y = x）前
#先执行control_dependencies中的内容（在这里就是 x_plus_1）
with tf.control_dependencies([x_plus_1]):
    y = x
    #y = tf.add(x , 0.0)
    #y = tf.identity(x)
init = tf.initialize_all_variables()
  
with tf.Session() as sess:
    init.run()

    
    writer = tf.summary.FileWriter("TensorBoard/output", sess.graph)
    for i in np.arange(5):
        print(y.eval())

    writer.close()