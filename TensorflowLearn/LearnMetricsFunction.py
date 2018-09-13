
import tensorflow as tf
import numpy as np
labels = np.array([[1,1,1,0],
                   [1,1,1,0],
                   [1,1,1,0],
                   [1,1,1,0]], dtype=np.uint8)

predictions = np.array([[1,0,0,0],
                        [1,1,0,0],
                        [1,1,1,0],
                        [0,1,1,1]], dtype=np.uint8)

n_batches = len(labels)


# Create running variables
N_CORRECT = 0
N_ITEMS_SEEN = 0

def reset_running_variables():
    """ Resets the previous values of running variables to zero """
    global N_CORRECT, N_ITEMS_SEEN
    N_CORRECT = 0
    N_ITEMS_SEEN = 0

def update_running_variables(labs, preds):
    global N_CORRECT, N_ITEMS_SEEN
    N_CORRECT += (labs == preds).sum()
    N_ITEMS_SEEN += labs.size

def calculate_accuracy():
    global N_CORRECT, N_ITEMS_SEEN
    return float(N_CORRECT) / N_ITEMS_SEEN

def batch_calculate():
    for i in range(n_batches):
        reset_running_variables()
        update_running_variables(labs=labels[i], preds=predictions[i])
        acc = calculate_accuracy()
        print("- [NP] batch {} score: {}".format(i, acc))

def mannual():
    n_items = labels.size
    accuracy = (labels ==  predictions).sum() / n_items
    print("Mannual Accuracy :", accuracy)


def tfacc():
    graph = tf.Graph()
    with graph.as_default():
        # Placeholders to take in batches onf data
        tf_label = tf.placeholder(dtype=tf.int32, shape=[None])
        tf_prediction = tf.placeholder(dtype=tf.int32, shape=[None])

        # Define the metric and update operations
        tf_metric, tf_metric_update = tf.metrics.accuracy(tf_label,
                                                          tf_prediction,
                                                          name="my_metric")

        # Isolate the variables stored behind the scenes by the metric operation
        running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="my_metric")

        # Define initializer to initialize/reset running variables
        running_vars_initializer = tf.variables_initializer(var_list=running_vars)


    with tf.Session(graph=graph) as session:
        session.run(tf.global_variables_initializer())

        # initialize/reset the running variables
        session.run(running_vars_initializer)

        for i in range(n_batches):
            # Update the running variables on new batch of samples
            feed_dict={tf_label: labels[i], tf_prediction: predictions[i]}
            session.run(tf_metric_update, feed_dict=feed_dict)
            
            # Calculate the score on this batch
            score = session.run(tf_metric)
            print("[TF] batch {} score: {}".format(i, score))

        # Calculate the score
        score = session.run(tf_metric)
        print("[TF] SCORE: ", score)

    with tf.Session(graph=graph) as session:
        session.run(tf.global_variables_initializer())

        for i in range(n_batches):
            # Reset the running variables
            session.run(running_vars_initializer)

            # Update the running variables on new batch of samples
            feed_dict={tf_label: labels[i], tf_prediction: predictions[i]}
            session.run(tf_metric_update, feed_dict=feed_dict)

            # Calculate the score on this batch
            score = session.run(tf_metric)
            print("[TF] batch {} score: {}".format(i, score))
        # Calculate the score
        score = session.run(tf_metric)
        print("[TF] SCORE: ", score)

if __name__ == '__main__': 
    #mannual()
    #batch_calculate()
    tfacc()
