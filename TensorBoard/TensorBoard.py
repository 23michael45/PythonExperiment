import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


learning_rate = 0.001
is_training = True
dropout = 0.25 # Dropout, probability to drop a unit
num_classes = 10 # MNIST total classes (0-9 digits)

data_path = './data/'
mnist = input_data.read_data_sets(data_path, one_hot=True)

def trainCnn():
    print("Train Cnn Start")
    with tf.name_scope('conv_layer'):

        x = tf.placeholder("float", [None, 28,28,1],name = "x")
        y_ = tf.placeholder("float", [None,10],name = "y_")

         # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer with 64 filters and a kernel size of 3
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        fc = tf.layers.dense(fc1, num_classes)
        #fc = tf.nn.relu(fc)
        y = tf.nn.softmax(fc)
        y = tf.identity(y, name="y")

        tf.summary.scalar('y', y)

        

        #choose one of below,they are equal
        loss_op = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),1))
        #loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fc, labels=tf.cast(tf.argmax(y_,1), dtype=tf.int32)))
        
        tf.summary.histogram('loss_op', loss_op)
        
        merged_summary_op = tf.summary.merge_all()
        
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_step = optimizer.minimize(loss_op,global_step=tf.train.get_global_step())

        # Evaluate the accuracy of the model
        accuracy, accuracy_op= tf.metrics.accuracy(labels=tf.argmax(y_,1), predictions=tf.argmax(y,1))
       
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())



        writer = tf.summary.FileWriter("TensorBoard/output", sess.graph)


        for i in range(100):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            batch_xs = np.reshape(batch_xs,(-1,28,28,1))
            [t,e,r] = sess.run([train_step,loss_op,y], feed_dict={x: batch_xs, y_: batch_ys})
            
            print("step:{} {}".format(i,e))

            sess.run(merged_summary_op)

            #print("Cnn Step:" + str(i) +  " Accuracy:" + str(result));


        testimages = mnist.test.images
        testimages = np.reshape(testimages,(-1,28,28,1))
        result = sess.run([accuracy,accuracy_op], feed_dict={x: testimages, y_: mnist.test.labels})
        print("Cnn Final Accuracy:" + str(result))

     
        writer.close()




if __name__ == '__main__': 
    mnist = trainCnn()
