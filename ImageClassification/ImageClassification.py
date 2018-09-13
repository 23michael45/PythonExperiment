import numpy as np
import tensorflow as tf
import time
from scipy.misc import imread
from scipy import misc  
import os
import cv2
from os import listdir, mkdir, sep
from os.path import join, exists, splitext
SrcDir = "Images/Cover"
PreprocessDir = "Images/Preprocess/"
ModelSaveDir = "Images/Model/"
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256


def list_images(directory):
    images = []
    for file in listdir(directory):
        name = file.lower()
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
    return images
#随机旋转图片  
def random_rotate_image_func(image):  
   #旋转角度范围  
   angle = np.random.uniform(low=-30.0, high=30.0)  
   return misc.imrotate(image, angle, 'bicubic')  
def random_rotate_image(image):
   image_rotate = tf.py_func(random_rotate_image_func, [image], tf.uint8) 
   return image_rotate


def preprocessone(image):


      # 神经网络输入的尺寸
      height = IMAGE_HEIGHT
      width = IMAGE_WIDTH

      with tf.Graph().as_default(): 
         
          rotate_image = random_rotate_image(image)
          # Image processing for training the network. Note the many random
          # distortions applied to the image.
          # 随机裁剪
          # Randomly crop a [height, width] section of the image.
          #crop_image = tf.random_crop(rotate_image, [height, width, 3])

          # Randomly flip the image horizontally.
          #flip_image = tf.image.random_flip_left_right(crop_image)


          #rotate_image = tf.cast(rotate_image,tf.uint8)
          
          # Because these operations are not commutative, consider randomizing
          # the order their operation.
          brightness_image = tf.image.random_brightness(rotate_image, max_delta=0.3)
          contrast_image = tf.image.random_contrast(brightness_image, lower=0.2, upper=1.8)

          # Subtract off the mean and divide by the variance of the pixels.
          #float_image = tf.image.per_image_standardization(contrast_image)

          #ret = contrast_image

          # Set the shapes of tensors.
          #float_image.set_shape([height, width, 3])
          #Fint_image = tf.cast(float_image, tf.uint8)

          jpg = tf.image.encode_jpeg(contrast_image, format='rgb', quality=100)

          with tf.Session() as sess:  
            sess.run(tf.global_variables_initializer())  
            sess.run(tf.local_variables_initializer())  
            result = sess.run(jpg)  
            
            return result
def preprocess():
    files = list_images(SrcDir)
    print(files)
    for file in files:
        
      path, filename = os.path.split(file)
      covername,ext = os.path.splitext(filename)
      dir = PreprocessDir+"/" + covername + "/";
      if not os.path.exists(dir):
          os.makedirs(dir)

    


      #file_contents = tf.read_file(file)  
      #image = tf.image.decode_image(file_contents, channels=3)  
      image = imread(file)

      for count in range(10):
          result = preprocessone(image)
          
          with open(dir  + str(count) + '.jpg','wb') as f:  
               f.write(result)  


def read_img(path):

    folders = os.listdir(path)
    cate=[path+x for x in  folders]
    cate=[f for f in cate if os.path.isdir(f)]
    imgs=[]
    labels=[]
    for idx,folder in enumerate(cate):
        for im in os.listdir(folder):
            fullpath = folder + '/' + im
            print('reading the images:%s'%(fullpath))
            img=imread(fullpath)
            img=cv2.resize(img,(100,100),interpolation=cv2.INTER_CUBIC)
            imgs.append(img)

            """
            arr = []

            for d in range(len(cate)):
                if d == idx:
                    arr.append(1)
                else:
                    arr.append(0)

            labels.append(arr)
            """
            labels.append(idx)
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)
def train():
    """
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    x_train = mnist.train.images # Returns np.array
    y_train = np.asarray(mnist.train.labels, dtype=np.int32)
    x_val = mnist.test.images # Returns np.array
    y_val = np.asarray(mnist.test.labels, dtype=np.int32)

    
    x_train = x_train.reshape([-1, 28, 28, 1])
    x_val = x_val.reshape([-1, 28, 28, 1])

    """
    
    data,label=read_img(PreprocessDir)
    #打乱顺序
    num_example=data.shape[0]
    arr=np.arange(num_example)
    np.random.shuffle(arr)
    data=data[arr]
    label=label[arr]

    #将所有数据分为训练集和验证集
    ratio=0.8
    s=np.int(num_example*ratio)
    x_train=data[:s]
    y_train=label[:s]
    x_val=data[s:]
    y_val=label[s:]
    

    #-----------------构建网络----------------------
    #占位符
    x=tf.placeholder(tf.float32,shape=[None,100,100,3],name='x')
    y_=tf.placeholder(tf.int32,shape=[None,],name='y_')


    #第一个卷积层（100——>50)
    conv1=tf.layers.conv2d(
          inputs=x,
          filters=32,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool1=tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    #第二个卷积层(50->25)
    conv2=tf.layers.conv2d(
          inputs=pool1,
          filters=64,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool2=tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

 
    #第三个卷积层(25->12)
    conv3=tf.layers.conv2d(
          inputs=pool2,
          filters=128,
          kernel_size=[3, 3],
          padding="same",
          activation=tf.nn.relu,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool3=tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    #keep_prob = tf.placeholder(tf.float32)
    pool3 = tf.nn.dropout(pool3,0.5 )
    """
    #第四个卷积层(12->6)
    conv4=tf.layers.conv2d(
          inputs=pool3,
          filters=128,
          kernel_size=[3, 3],
          padding="same",
          activation=tf.nn.relu,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool4=tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
    """

    re1 = tf.reshape(pool3, [-1, 12 * 12 * 128])

    #全连接层
    dense1 = tf.layers.dense(inputs=re1, 
                          units=1024, 
                          activation=tf.nn.relu,
                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    
    dense1 = tf.nn.dropout(dense1,0.5 )
    dense2= tf.layers.dense(inputs=dense1, 
                          units=512, 
                          activation=tf.nn.relu,
                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    
    #dense2 = tf.nn.dropout(dense2,0.5 )
    logits= tf.layers.dense(inputs=dense2, 
                            units=10, 
                            activation=None,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    #---------------------------网络结束---------------------------

    loss=tf.losses.sparse_softmax_cross_entropy(labels=y_,logits=logits)
    train_op=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    correct_prediction = tf.equal(tf.cast(tf.argmax(logits,1),tf.int32), y_)    
    acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    #训练和测试数据，可将n_epoch设置更大一些
    saver = tf.train.Saver()
    n_epoch=10
    batch_size=64
    sess=tf.InteractiveSession()  
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epoch):
        start_time = time.time()
    
        #training
        train_loss, train_acc, n_batch = 0, 0, 0
        for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
            _,err,ac=sess.run([train_op,loss,acc], feed_dict={x: x_train_a, y_: y_train_a})
            #print(sess.run([logits,loss], feed_dict={x: x_train_a, y_: y_train_a}))

            train_loss += err; train_acc += ac; n_batch += 1
        print("   train loss: %f" % (train_loss/ n_batch))
        print("   train acc: %f" % (train_acc/ n_batch))
    
        #validation
        val_loss, val_acc, n_batch = 0, 0, 0
        for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
            err, ac = sess.run([loss,acc], feed_dict={x: x_val_a, y_: y_val_a})
            val_loss += err; val_acc += ac; n_batch += 1
        print("   validation loss: %f" % (val_loss/ n_batch))
        print("   validation acc: %f" % (val_acc/ n_batch))
        print("-------------------------------------------------------------------------------------------------------")

    save_path = saver.save(sess, ModelSaveDir + "cnnmodel.ckpt")
    sess.close()


#定义一个函数，按批次取数据
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

def main():
    #preprocess()
    train()


if __name__ == '__main__':
    main()