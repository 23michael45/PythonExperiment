# -*- coding：utf-8 -*-

import tensorflow as tf
import skimage.io
import glob
import os
import numpy as np
import sys




def Resize(srcPath,dstPath,size):
    imgPaths=os.listdir(srcPath)
    
    for root, subFolder, files in os.walk(srcPath):
        for file in files:
            img = skimage.io.imread(os.path.join(root,file))
            height,width,channel = img.shape
            if width > height:
                img = img[ : ,width//2 - height//2 : width//2 + height//2  ]
                #skimage.io.imshow(img)
                #skimage.io.show()
                s = np.array(size)
                img = skimage.transform.resize(img,size)

                finalPath =  root.replace(srcPath,dstPath)
                
                try:
                    os.stat(finalPath)
                except:
                    os.mkdir(finalPath) 
                skimage.io.imsave(os.path.join(finalPath,file),img)
            elif width == height:
                s = np.array(size)
                img = skimage.transform.resize(img,size)
                
                finalPath =  root.replace(srcPath,dstPath)
                
                try:
                    os.stat(finalPath)
                except:
                    os.mkdir(finalPath) 
                skimage.io.imsave(os.path.join(finalPath,file),img)



#随机旋转图片
import time
from scipy import misc
from random import randint
def random_rotate_image_func(image):
    #旋转角度范围
    angle = np.random.uniform(low=-30.0, high=30.0)
    return misc.imrotate(image, angle, 'bicubic')


def random_image(image_file):
    with tf.Graph().as_default():

        tf.set_random_seed(time.time())

        file_contents = tf.read_file(image_file)

        image = tf.image.decode_image(file_contents, channels=3)

        #image = tf.image.random_flip_left_right(image)            
        image = tf.image.random_brightness(image, max_delta=0.3)
        
        image = tf.image.random_contrast(image, lower=0, upper=4)
        #image = tf.image.random_hue(image, 0.5)
        #image = tf.image.random_saturation(image, lower=0, upper=2)

        image = tf.py_func(random_rotate_image_func, [image], tf.uint8)

        s = randint(128,256)
        image = tf.random_crop(image, [s, s, 3])

        imgjpg = (tf.image.encode_jpeg(image))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            results = sess.run(imgjpg)

            #for idx,ret in enumerate(results):
                #with open('data/'+str(idx)+'.png','wb') as f:
                    #f.write(ret)
            return results



def ProcessImgs(srcPath,dstPath):
    imgPaths=os.listdir(srcPath)
    
    num = 500
    for imgPath in imgPaths:  
        srcFile = os.path.join(srcPath,imgPath)
        dstFileDir = os.path.join(dstPath,os.path.splitext(imgPath)[0])

        for i in np.arange(num):
            dstFile = os.path.join(dstFileDir,"{}.jpg".format(i))
            ret = random_image(srcFile)


            try:
                os.stat(dstFileDir)
            except:
                os.mkdir(dstFileDir) 
            with open(dstFile,'wb') as f:
                f.write(ret)


def Train(srcPath,train = True):
    
    # 模型文件路径
    model_path = "model/image_model"

    dic = {}
    for root, subFolder, files in os.walk(srcPath):
        for item in files:
            if(not root in dic):
                dic[root] = []
            dic[root].append(item)
            #print("{} {} {}".format(root,subFolder,item))
    # 计算有多少类图片
    num_classes = len(dic)


    datas = []
    labels = []

    learning_rate=1e-3
    class_data_len = 500

    n_epoch=50                                                                                                               
    batch_size=40
    train_num_start = 0
    train_num_end = 10
    validate_num_start = 460
    validate_num_end = 500


    i = 0
    for key,value in dic.items():
        for file in value:
             img = skimage.io.imread(os.path.join(key,file))
             datas.append(img / 255.0)
             labels.append(i)
        i += 1   

        
    datas = np.array(datas)
    labels = np.array(labels)

    
    
    datas_train = np.empty((0,256,256,3))
    labels_train = np.empty((0,))
    datas_validate = np.empty((0,256,256,3))
    labels_validate = np.empty((0,))
    for i in np.arange(num_classes):
        start = train_num_start + i *class_data_len
        end = train_num_end + i *class_data_len
        datas_train = np.concatenate((datas_train,datas[start:end]))
        labels_train = np.concatenate((labels_train,labels[start:end]))

        start = validate_num_start + i *class_data_len
        end = validate_num_end + i *class_data_len
        datas_validate = np.concatenate((datas_validate,datas[start:end]))
        labels_validate = np.concatenate((labels_validate,labels[start:end]))
    


        
    num_examples = datas_train.shape[0]
    indices = np.arange(num_examples)
    np.random.shuffle(indices)
    datas_train = datas_train[indices]
    labels_train = labels_train[indices]

    


    # 定义Placeholder，存放输入和标签
    datas_placeholder = tf.placeholder(tf.float32, [None, 256, 256, 3],name='Input')
    labels_placeholder = tf.placeholder(tf.int32, [None])
    


    # 存放DropOut参数的容器，训练时为0.25，测试时为0
    dropout_placeholdr = tf.placeholder(tf.float32)


    # 定义卷积层, 20个卷积核, 卷积核大小为5，用Relu激活
    conv0 = tf.layers.conv2d(datas_placeholder, 32, 5, activation=tf.nn.relu)
    # 定义max-pooling层，pooling窗口为2x2，步长为2x2
    pool0 = tf.layers.max_pooling2d(conv0, [2, 2], [2, 2])

    # 定义卷积层, 40个卷积核, 卷积核大小为4，用Relu激活
    conv1 = tf.layers.conv2d(pool0, 48, 4, activation=tf.nn.relu)
    # 定义max-pooling层，pooling窗口为2x2，步长为2x2
    pool1 = tf.layers.max_pooling2d(conv1, [2, 2], [2, 2])

    # 定义卷积层, 80个卷积核, 卷积核大小为4，用Relu激活
    conv2 = tf.layers.conv2d(pool1, 64, 4, activation=tf.nn.relu)
    # 定义max-pooling层，pooling窗口为2x2，步长为2x2
    pool2 = tf.layers.max_pooling2d(conv2, [2, 2], [2, 2])


    # 将3维特征转换为1维向量
    flatten = tf.layers.flatten(pool2)

    # 全连接层，转换为长度为100的特征向量
    fc = tf.layers.dense(flatten, 100, activation=tf.nn.relu)

    # 加上DropOut，防止过拟合
    dropout_fc = tf.layers.dropout(fc, dropout_placeholdr)

    # 未激活的输出层
    logits = tf.layers.dense(dropout_fc, num_classes)

    predicted_labels = tf.arg_max(logits, 1,name='Output')


    # 利用交叉熵定义损失
    losses = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.one_hot(labels_placeholder, num_classes),
        logits=logits
    )
    # 平均损失
    mean_loss = tf.reduce_mean(losses)

    # 定义优化器，指定要优化的损失函数
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(losses)

    
    mean_loss_summary = tf.summary.scalar("training_mean_loss", mean_loss)
    #validation_summary = tf.summary.scalar("validation_accuracy", accuracy)
    
    summary = tf.summary.merge_all()
    # 用于保存和载入模型
    saver = tf.train.Saver()

    with tf.Session() as sess:
        

        summary_writer = tf.summary.FileWriter("logs/", sess.graph) #第一个参数指定生成文件的目录
        if train:

            print("训练模式")
            # 如果是训练，初始化参数
            sess.run(tf.global_variables_initializer())
            # 定义输入和Label以填充容器，训练时dropout为0.25
            
            for epoch in range(n_epoch):

                iter_num = int((train_num_end - train_num_start) * num_classes / batch_size)
                for iter in range(iter_num):
                    start = iter * batch_size
                    end = iter * batch_size + batch_size
                    train_feed_dict = {
                    datas_placeholder: datas_train[start:end],
                    labels_placeholder: labels_train[start:end],
                    dropout_placeholdr: 0.25}


                    _, mean_loss_val,summary_str = sess.run([optimizer, mean_loss,summary], feed_dict=train_feed_dict)

                    summary_writer.add_summary(summary_str, epoch)
                    summary_writer.flush()

                    if epoch % 10 == 0:
                        print("step = {}\tmean loss = {}".format(epoch, mean_loss_val))
            saver.save(sess, model_path)
            print("训练结束，保存模型到{}".format(model_path))



            # label和名称的对照关系
            label_name_dict = {
                0: "玛莎拉蒂",
                1: "奔    驰",
                2: "凯迪拉克",
                3: "宝    马"
            }
            # 定义输入和Label以填充容器，测试时dropout为0
            test_feed_dict = {
                datas_placeholder: datas_validate,
                labels_placeholder: labels_validate,
                dropout_placeholdr: 0
            }
            predicted_labels_val = sess.run(predicted_labels, feed_dict=test_feed_dict)
            # 真实label与模型预测label
            for real_label, predicted_label in zip(labels_validate, predicted_labels_val):
                # 将label id转换为label名
                real_label_name = label_name_dict[real_label]
                predicted_label_name = label_name_dict[predicted_label]                
                print("t{} => {}".format(real_label_name, predicted_label_name))
            
            equalArray =  (labels_validate == predicted_labels_val).astype(int) 


            accuracy = np.mean(equalArray)
            print('accuracy:' + str(accuracy))

def Inference(path):
    # label和名称的对照关系
    label_name_dict = {
        0: "玛莎拉蒂",
        1: "奔    驰",
        2: "凯迪拉克",
        3: "宝    马"
    }
    
    img = skimage.io.imread(path)
    skimage.io.imshow(img)
    skimage.io.show()
    print("测试模式")
    # 如果是测试，载入参数
    
    sess = tf.Session()
    model_file=tf.train.latest_checkpoint('model/')
    saver = tf.train.import_meta_graph(model_file + '.meta')
    saver.restore(sess, model_file)
    print("从{}载入模型".format(model_file))
    
    graph = tf.get_default_graph()
    with graph.as_default():
        xs = graph.get_tensor_by_name('Input:0')
        ret = graph.get_tensor_by_name('Output:0')


        datax = np.empty((0,))
        datax = img[np.newaxis,...]
    
        prediction_value = sess.run(ret, feed_dict={xs: datax})  
        
        for prediction in prediction_value:
            real_label_name = label_name_dict[prediction]
            print("结果:" + real_label_name)





if __name__ == '__main__':
    
    srcPath = 'Image/Logo'
    dstPath = 'Image/LogoResize'
    postPath = 'Image/LogoPost'
    finalPath = 'Image/LogoFinal'

    #Resize(srcPath,dstPath,(256,256))

    #ProcessImgs(dstPath,postPath)
    #Resize(postPath,finalPath,(256,256))

    Train(finalPath)

    #Inference(os.path.join(finalPath,'2/337.jpg'))
    #Train(finalPath,False);