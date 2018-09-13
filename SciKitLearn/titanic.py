
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn import tree
import tensorflow as tf
import numpy as np


#features=['Pclass','Sex','Age','SibSp','Parch']
#features=['Pclass','Sex','Age','SibSp','Parch','Embarked']
#features=['Pclass','Sex']
#features=['Age','Sex']

features=['Pclass','Sex','Age','SibSp','Parch','Fare']
def loadTrainData():
    df=pd.read_csv('data/titanic_all/train.csv')
    print(df.head())
    print(df.info())

    print(df.Age.mean()) 
    print(df.Age.median())

    df.Age.fillna(df.Age.mean(),inplace=True) 
    #data.fillna会返回新的对象，如果在原有列上把空值进行填充，需要添加参数inplace=True

    df.Embarked.fillna('S',inplace=True) 
    #选取Pclass，Sex，Age，SibSp，Parch五个特征

    from sklearn.preprocessing import LabelEncoder
    le=LabelEncoder()
    le.fit(df['Sex'])
    #用离散值转化标签值
    df['Sex']=le.transform(df['Sex'])
    #print(df.Sex)
    
    sex_pivot = df.pivot_table(index="Sex",values="Survived")
    print(sex_pivot)

    le=LabelEncoder()
    le.fit(df['Embarked'])
    #用离散值转化标签值
    df['Embarked']=le.transform(df['Embarked'])


    X=df[features]
    y=df['Survived']
    print(y)

    y = y.values
   
    return X.values,y
def loadTestData(features):
    test=pd.read_csv('data/titanic_all/test.csv')
    test.Age.fillna(test.Age.mean(),inplace=True) 
    from sklearn.preprocessing import LabelEncoder
    le=LabelEncoder()
    le.fit(test['Sex'])
    #用离散值转化标签值
    test['Sex']=le.transform(test['Sex'])


    le.fit(test['Embarked'])
    #用离散值转化标签值
    test['Embarked']=le.transform(test['Embarked'])


    X_test=test[features]
    return X_test.values,test
def saveCSV(result,csv):
   
    csv.insert(1,'Survived',result)
    final=csv.iloc[:,0:2]
    final.to_csv('data/titanic_all/final.csv',index=False)
def decisionTree():

    X , y = loadTrainData()

    dt=tree.DecisionTreeClassifier()
    dt=tree.DecisionTreeClassifier('entropy')

    score=cross_val_score(dt,X,y,cv=5,scoring='accuracy')
    import numpy as np
    print(np.mean(score))


    dt.fit(X,y)
    from sklearn.tree import export_graphviz  #通过graphviz绘制决策树
    with open('data/titanic_all/titanic.dot','w')as f:
        f=export_graphviz(dt,feature_names=['Pclass','Sex','Age','SibSp','Parch'],out_file=f)
     #export_graphviz第一个参数填入决策树的模型，feature_names填入参与的特征名，out_file即指定输出文件

    X_test ,csv=  loadTestData(features)
    result=dt.predict(X_test)
    saveCSV(result,csv)


  



def fullConnection():
    
    inputData , inputLabel = loadTrainData()          
    n_values = np.max(inputLabel) + 1
    inputLabel = np.eye(n_values)[inputLabel]

    x_input=tf.placeholder(tf.float32,shape=[None,len(features)],name='x')

    #dense = tf.layers.dense(inputs=x_input, units=10, activation=None)
    #dense = tf.layers.dense(inputs=dense, units=10, activation=None)
    
    #dense = tf.nn.dropout(dense, 0.5)
    #dense = tf.layers.dense(inputs=dense, units=1024, activation=None)
    #dense = tf.layers.dense(inputs=dense, units=128, activation=None)
    #dense = tf.layers.dense(inputs=dense, units=32, activation=None)
    logits = tf.layers.dense(inputs=x_input, units=2, activation=tf.nn.softmax)

 
    inputLabel = np.int32(inputLabel)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(inputLabel * tf.log(logits)))

    #cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels = inputLabel, logits=logits)

    #labelout = tf.reduce_max(logits)
    #loss = tf.reduce_mean(logits - inputLabel)
    
    train_op=tf.train.AdamOptimizer(learning_rate=0.01).minimize(cross_entropy)

    
    maxIndex = tf.cast(tf.argmax(logits,axis = 1),tf.int64)
    maxLable =  tf.argmax(inputLabel,axis = 1)

    correct_prediction = tf.equal(maxIndex,maxLable)  
    acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    # 初始化所有变量
    init = tf.global_variables_initializer()
    # 激活会话
    with tf.Session() as sess:
            sess.run(init)

            # 迭代次数 = 10000
            for i in range(100):
                # 训练
                sess.run(train_op ,feed_dict={x_input: inputData})
                #print(sess.run([maxIndex,maxLable],feed_dict={x_input: inputData}))
                #print(sess.run([acc],feed_dict={x_input: inputData}))



            X_test ,csv=  loadTestData(features)
            
            result = sess.run(logits ,feed_dict={x_input: X_test})
            
            result = np.argmax(result,axis = 1)
            saveCSV(result,csv)



            

def tftest():
    data = [[1,0],
            [1,0],
            [0,1],
            [1,0],
            [0,1]]

    datax = [1,2,3,4,5]
    datay = [[2,1],
            [1,100]]

    x = tf.Variable(data)
    y = tf.argmax(x,dimension=1)

    mean = tf.reduce_mean(datay);

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
            sess.run(init)
            logits = [[0.5,0.7,0.3],[0.8,0.2,0.9]]
            likeones =  tf.ones_like(logits)
            loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = logits,labels = likeones)
            print(sess.run([loss,likeones]))

if __name__ == '__main__':
    #tftest()
    fullConnection()
    #decisionTree()
