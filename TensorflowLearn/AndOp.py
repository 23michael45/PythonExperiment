import tensorflow as tf
import numpy as np


data = [[1,1,1,0],
        [1,0,0,1],
        [0,1,0,1],
        [0,0,0,1]]

datax = np.float32([d[0:2] for d in data]);
label = np.float32([d[2:4] for d in data]);


#datax = np.transpose(datax)

print(datax)
print(label)


def prediction():

   logits = tf.nn.xw_plus_b(xs,weights , biases)

   return logits
def predictionlayer():
    dense1 = tf.layers.dense(inputs=xs, units=2, activation=tf.nn.softmax)
    #dense2= tf.layers.dense(inputs=dense1, units=1, activation=None)
    
    #y = tf.nn.softmax(tf.matmul(xs,weights) + biases)
    return dense1


def train():
    
    xs = tf.placeholder(tf.float32, [None, 2],name = 'InputX')
    ys = tf.placeholder(tf.float32, [None, 2])


    weights = tf.Variable(tf.random_uniform([2, 2], -1.0, 1.0),name = 'weights',dtype = tf.float32)
    biases = tf.Variable(tf.zeros([2]),name = 'bias',dtype = tf.float32)
    #with tf.variable_scope('and'):
  
    #logits = predictionlayer()
    logits = tf.layers.dense(inputs=xs, units=6, activation=None)
    logits = tf.layers.dense(inputs=logits, units=2, activation=None)

    logitsA = tf.layers.dense(inputs=xs, units=2, activation=tf.nn.softmax)

    logitsB = tf.nn.softmax(logits,name = "LogitsResult")

    #cross_entropy = tf.reduce_mean(-tf.reduce_sum(label * tf.log(logitsB), reduction_indices=[1])) 

    
    cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=label, logits=logits)
    cross_entropy =  tf.reduce_mean(tf.reduce_sum(cross_entropy))

    #with tf.name_scope('loss'):
    #    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
    #        labels=label, logits=logits)
    #cross_entropy = tf.reduce_mean(cross_entropy)
    #loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - logits)))

    train_step = tf.train.AdamOptimizer(0.1).minimize(cross_entropy)

    # 初始化所有变量
    init = tf.global_variables_initializer()
    # 激活会话
    #with tf.Graph().as_default():
    with tf.Session() as sess:
        sess.run(init)

        #print(sess.run(weights))

        # 迭代次数 = 10000
        for i in range(500):
            # 训练
            sess.run(train_step, feed_dict={xs: datax, ys: label})
            #print(sess.run(cross_entropy,feed_dict={xs: datax, ys: label}))
            #print(sess.run([weights,biases]))
            
        #print(sess.run(logits,feed_dict={xs: datax, ys: label}))
        #print(sess.run(logitsA,feed_dict={xs: datax, ys: label}))
        #print(sess.run(logitsB,feed_dict={xs: datax, ys: label}))

        saver = tf.train.Saver()
        saver.save(sess,'ckpt/andop.ckpt')

            
        graph = tf.get_default_graph()
        for op in graph.get_operations():
            print(str(op.name))
        print('----------------------------------------------------------------------------------------------------------------------------------------')

        prediction_value = sess.run(logitsB, feed_dict={xs: datax})  
        print(prediction_value)
        print(np.around(prediction_value))

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
def inference():
    sess = tf.Session()

    model_file=tf.train.latest_checkpoint('ckpt/')
    
    #saver = tf.train.Saver()
    saver = tf.train.import_meta_graph(model_file + '.meta')
    saver.restore(sess,model_file)

    graph = tf.get_default_graph()
    with graph.as_default():
        for op in graph.get_operations():
            print(str(op.name))
    print('----------------------------------------------------------------------------------------------------------------------------------------')
    #for tensor in tf.get_default_graph().get_tensors():
    #    print(str(tensor.name))
    #print_tensors_in_checkpoint_file(model_file,None,True,True)

    #ret = graph.get_operation_by_name('and/LogitsResult')
    
    
    #xs = graph.get_operation_by_name('InputX:0')
    
    xs = graph.get_tensor_by_name('InputX:0')
    ret = graph.get_tensor_by_name('LogitsResult:0')
    datax = np.float32([d[0:2] for d in data]);
    
    prediction_value = sess.run(ret, feed_dict={xs: datax})  
    print(prediction_value)
    print(np.around(prediction_value))

    
from tensorflow.python.framework import graph_util
def SavePB():
    
    sess = tf.Session()

    model_file=tf.train.latest_checkpoint('ckpt/')
    
    #saver = tf.train.Saver()
    saver = tf.train.import_meta_graph(model_file + '.meta')
    saver.restore(sess,model_file)
    
    graph = tf.get_default_graph()



    # convert_variables_to_constants 需要指定output_node_names，list()，可以多个
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['InputX','LogitsResult'])

    tf.train.write_graph(sess.graph_def, 'ckpt/', 'andop.pbtxt')
    # 测试 OP 

    # 写入序列化的 PB 文件
    with tf.gfile.FastGFile('ckpt/andop.pb', mode='wb') as f:
        f.write(constant_graph.SerializeToString())

if __name__ == '__main__':
    #train()
    inference()
    #SavePB()