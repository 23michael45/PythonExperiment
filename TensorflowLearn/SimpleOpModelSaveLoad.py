import tensorflow as tf
import numpy as np


#data = [[1,1,1,0],
#        [1,0,0,1],
#        [0,1,0,1],
#        [0,0,0,1]]

data = [[1,1,0,1],
        [1,0,1,0],
        [0,1,1,0],
        [0,0,1,0]]

datax = np.float32([d[0:2] for d in data]);
label = np.float32([d[2:4] for d in data]);


#datax = np.transpose(datax)

print(datax)
print(label)


def train():
    
    xs = tf.placeholder(tf.float32, [None, 2],name = 'InputX')
    ys = tf.placeholder(tf.float32, [None, 2])

    logits = tf.layers.dense(inputs=xs, units=6, activation=None,name = "Dense1")

    logits = tf.layers.dense(inputs=logits, units=2, activation=None,name = "Dense2")

    logits = tf.nn.softmax(logits,name = "LogitsResult")

    cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=label, logits=logits)

    #cross_entropy =  tf.reduce_mean(tf.reduce_sum(cross_entropy))
    
    cross_entropy =  tf.reduce_mean(cross_entropy)

    train_step = tf.train.AdamOptimizer(0.1).minimize(cross_entropy)

    # 初始化所有变量
    init = tf.global_variables_initializer()

    # 激活会话
    #with tf.Graph().as_default():

    with tf.Session() as sess:

        
        summary_writer = tf.summary.FileWriter("logs/Op", sess.graph) #第一个参数指定生成文件的目录

        sess.run(init)

        
        graph = tf.get_default_graph()
        kernel_1 = graph.get_tensor_by_name('Dense1/kernel:0')       
        bias_1 = graph.get_tensor_by_name('Dense1/bias:0')
        kernel_2 = graph.get_tensor_by_name('Dense2/kernel:0')
        bias_2 = graph.get_tensor_by_name('Dense2/bias:0')

        # 迭代次数 = 10000
        for i in range(500):
            # 训练
            #sess.run(train_step, feed_dict={xs: datax, ys: label})
            _,k_1,k_2,b1,b2 = sess.run([train_step,kernel_1,kernel_2,bias_1,bias_2], feed_dict={xs: datax, ys: label})
            #print(k_1)
            #print(k_2)
            #print(b1)
            #print(b2)
            #print(sess.run(cross_entropy,feed_dict={xs: datax, ys: label}))
            #print(sess.run([weights,biases]))
            
        #print(sess.run(logits,feed_dict={xs: datax, ys: label}))

        saver = tf.train.Saver()
        saver.save(sess,'ckpt/andop.ckpt')

            
        for op in graph.get_operations():
            print(str(op.name))
        print('----------------------------------------------------------------------------------------------------------------------------------------')

        prediction_value = sess.run(logits, feed_dict={xs: datax})  
        print(prediction_value)
        print(np.around(prediction_value))
        print(np.argmax(prediction_value,axis = 1))

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
