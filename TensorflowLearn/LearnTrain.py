from os import listdir
from os.path import isfile, join
import tensorflow as tf
import numpy as np

path = 'data'

filenames = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
filenames = sorted(filenames) 
filename_queue = tf.train.string_input_producer(filenames,shuffle = False,num_epochs=5) 

images = ['img1', 'img2', 'img3', 'img4', 'img5']  
labels= [1,2,3,4,5]  
f_queue = tf.train.slice_input_producer([images, labels],num_epochs=5,shuffle=False)  


#filename_queue_ = tf.train.slice_input_producer([filenames],num_epochs=5,shuffle=False)  
 

#filenames = tf.constant(filenames)


#reader = tf.TextLineReader()
#key, value = reader.read(filename_queue)

# reader从文件名队列中读数据。对应的方法是reader.read
reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)

batchkey,batchvalue = tf.train.batch([key,value], 12, dynamic_pad=True)

rd = tf.random_uniform((4, 4), minval=256, maxval=512 + 1, dtype=tf.int32)


with tf.Session() as sess:
  tf.local_variables_initializer().run()
  # Start populating the filename queue.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  print(sess.run(rd))

  i = 0
  while True:
        try:
            i += 1
            # 获取图片数据并保存
            #key_data,image_data = sess.run([key,value])
            #print('{} : {}'.format(i,key_data))

            f = sess.run(batchkey)
            print('{} : {}'.format(i,f))
        except Exception:
            break


  coord.request_stop()
  coord.join(threads)

