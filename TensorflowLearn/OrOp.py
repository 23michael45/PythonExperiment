import numpy as np
import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants
print(tf.__version__) 

data = [[1,1,1],
        [1,0,1],
        [0,1,1],
        [0,0,0]]
x_train = np.float32([d[0:2] for d in data]);
label = np.int32([d[2:3] for d in data]);


x_input = tf.placeholder(tf.float32, [None, 2],name='input')



def train_save():

    with tf.variable_scope('or'):
  
        logits = tf.layers.dense(inputs=x_input, units=10, activation=None)
        #logits = tf.layers.dense(inputs=logits, units=50, activation=None)
        logits = tf.layers.dense(inputs=logits, units=2, activation=None)
      
        logitsB = tf.nn.softmax(logits,name = "logits")  
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels = label, logits=logits)
    
        train_op=tf.train.AdamOptimizer(learning_rate=0.1).minimize(cross_entropy)


        correct_prediction = tf.equal(tf.cast(tf.argmax(logits,1),tf.int32), label)    
        acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


        # 初始化所有变量
        init = tf.global_variables_initializer()
        # 激活会话
        with tf.Session() as sess:
            sess.run(init)

            # 迭代次数 = 10000
            for i in range(50):
                # 训练
                sess.run(train_op, feed_dict={x_input: x_train})
                print(sess.run(cross_entropy,feed_dict={x_input: x_train}))
            
            print(np.around(sess.run(logitsB,feed_dict={x_input: x_train})))0
            print(logitsB.name)

            graph = convert_variables_to_constants(sess, sess.graph_def, ["or/logits"])
            tf.train.write_graph(graph, '.', 'graph.pb', as_text=False)
            tf.train.write_graph(graph, '.', 'graph_astext.pb', as_text=True)

with tf.Session() as sess:
    with open('./graph.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read()) 
        output = tf.import_graph_def(graph_def, input_map={'input:0':x_train},return_elements=['or/logits:0']) 

        
        print("Ops----------------------------------------------------------------------------------")
        op = sess.graph.get_operations()
        for n in op:

            print(n.name)
        
        print("----------------------------------------------------------------------------------")



        input = sess.graph.get_tensor_by_name('input:0')
        logits = sess.graph.get_tensor_by_name('import/or/logits:0')
        
        print(sess.run(input,feed_dict={input: x_train}))
        print("output: {0}".format(np.round(sess.run(output))))

        print("logits: {0}".format(np.round(sess.run(logits))))
        tflite_model =  tf.contrib.lite.toco_convert(graph_def, [], [output]) #这里[input], [out]这里分别是输入tensor或者输出tensor的集合,是变量实体不是名字
        open("model.tflite", "wb").write(tflite_model)

    print("----------------------------------------------------------------------------------")
    with open('./graph_astext.pb', 'r') as f:
        graph_def = tf.GraphDef()
        # 不使用graph_def.ParseFromString(f.read())
        from google.protobuf import text_format
        text_format.Merge(f.read(), graph_def)
        output = tf.import_graph_def(graph_def, input_map={'input:0':x_train},return_elements=['or/logits:0']) 
       
        print(np.round(sess.run(output)))


        
