# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 13:26:12 2017

@author: Alphatao
"""

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

#xtr 已有的数据， xte 待测试数据
xtr = tf.placeholder(tf.float32,[None,784])
xte = tf.placeholder(tf.float32,[784])

#距离L2 欧几里得距离
distance = tf.sqrt(tf.reduce_sum(tf.pow(tf.add(xtr,tf.negative(xte)),2), reduction_indices=1))

#挑选最近的一个
pred = tf.argmin(distance, 0)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

right = 0

train_x,train_y = mnist.train.next_batch(10000)
test_x,test_y = mnist.test.next_batch(1000)
for i in range(1000):
    ans = sess.run(pred, feed_dict = {xtr:train_x,xte:test_x[i,:]})
    print('#%d'%i,'pred is ',np.argmax(train_y[ans]),'true value is ',np.argmax(test_y[i]))
    if np.argmax(train_y[ans]) == np.argmax(test_y[i]):
        right +=1.0
acc = right/1000.0
print('acc=',acc)