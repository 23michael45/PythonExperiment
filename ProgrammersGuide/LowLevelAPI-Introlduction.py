from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

a = tf.constant(3.0, dtype=tf.float32,name = "a")
b = tf.constant(4.0) # also tf.float32 implicitly
total = a + b



print(a)
print(b)
print(total)

sess = tf.Session()
print(sess.run(total))

print(sess.run({'ab':(a, b), 'total':total}))




vec = tf.random_uniform(shape=(3,))
out1 = vec + 1
out2 = vec + 2
print(sess.run(vec))
print(sess.run(vec))
print(sess.run((out1, out2)))


x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = x + y


print(sess.run(z, feed_dict={x: 3, y: 4.5}))
print(sess.run(z, feed_dict={x: [1, 3], y: [2, 4]}))


print('-------------------------------------------------------------------------------------')

my_data = [
    [0, 1,],
    [2, 3,],
    [4, 5,],
    [6, 7,],
]
slices = tf.data.Dataset.from_tensor_slices(my_data)
next_item = slices.make_one_shot_iterator().get_next()

while True:
  try:
    print(next_item)
    print(sess.run(next_item))
  except tf.errors.OutOfRangeError:
    break


print('-------------------------------------------------------------------------------------')

x = tf.placeholder(tf.float32, shape=[None, 3])
linear_model = tf.layers.Dense(units=5)
y = linear_model(x)

init = tf.global_variables_initializer()
print(sess.run(init))

print(sess.run(y, {x: [[1, 2, 3],[4, 5, 6]]}))


print('-------------------------------------------------------------------------------------')

features = {
    'sales' : [[5], [10], [8], [9]],
    'department': ['sports', 'sports', 'gardening', 'gardening']}

department_column = tf.feature_column.categorical_column_with_vocabulary_list(
        'department', ['sports', 'gardening'])


print(department_column)
department_column = tf.feature_column.indicator_column(department_column)

print(department_column)

columns = [
    tf.feature_column.numeric_column('sales'),
    department_column
]

inputs = tf.feature_column.input_layer(features, columns)

var_init = tf.global_variables_initializer()
table_init = tf.tables_initializer()
sess = tf.Session()
sess.run((var_init, table_init))

print(sess.run(inputs))



print('-------------------------------------------------------------------------------------')

x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

linear_model = tf.layers.Dense(units=1)
y_pred = linear_model(x)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(y_pred))


loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
print(sess.run(loss))


optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)


for i in range(1000):
  _, loss_value = sess.run((train, loss))
  print(loss_value)


print(sess.run(y_pred))


saver = tf.train.Saver()
path = saver.save(sess, "dens-model")