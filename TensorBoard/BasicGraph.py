import tensorflow as tf
import numpy as np

#Here we are defining the name of the graph, scopes A, B and C.
with tf.name_scope("MyOperationGroup"):
    with tf.name_scope("Scope_A"):
        a = tf.add(1, 2, name="Add_these_numbers")
        b = tf.multiply(a, 3)
    with tf.name_scope("Scope_B"):
        c = tf.add(4, 5, name="And_These_ones")
        d = tf.multiply(c, 6, name="Multiply_these_numbers")

with tf.name_scope("Scope_C"):
    e = tf.multiply(4, 5, name="B_add")
    f = tf.div(c, 6, name="B_mul")


# 创建变量，保存计算结果
g = tf.Variable(0, dtype=tf.int64)
h = tf.assign_add(g,1)


tf.summary.scalar("g",g)
#tf.summary.scalar("h",h)
merged_summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for i in np.arange(10):
        writer = tf.summary.FileWriter("TensorBoard/output", sess.graph)
        
        
        # gvalue = g.eval()
        # print(gvalue)

        # update = tf.assign(g, gvalue)

        # print(sess.run(update))
        
        print(g.eval())
        print(sess.run(h))
        



        summary = sess.run(merged_summary_op)
        writer.add_summary(summary,i)
        writer.close()
