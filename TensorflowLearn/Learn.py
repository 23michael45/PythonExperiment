
import tensorflow as tf
import numpy as np



def Y():
    for i in range(5):
        print(i)
        yield i;

def main():
    y = Y()

    for r in y:
        print(r * 2)
   # for i in range(5):
   #     print(y.next())

    
    func = lambda x,y : x + y
    m = map(func,(1,3),(10,20))
    print(list(m))


    weight = [[1],[10],[100]]
    weight2 = [[1,10,100]]
        
    g = tf.Graph()
    with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
        input = tf.placeholder(tf.float32, shape=(1,3))

        w = tf.constant(weight,dtype = np.float32)
        w2 = tf.constant(weight2,dtype = np.float32)
        b = tf.constant([99],dtype = np.float32)
        product = tf.matmul(input, w) + b

        ret = product.eval(feed_dict={input:[[1,2,3]]})
       # ret = product.eval()
        print(ret)


if __name__ == '__main__':
    main()
    
    input("Press Enter to terminate")
