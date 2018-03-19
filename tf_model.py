# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.examples.tutorials.mnist import input_data


def initialization():
    # 进行placeholder等初始化
    # 我们feed的内容，用placeholder，将会传入参数，其他的初始化的变量用variable
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y_ = tf.placeholder("float", [None, 10])
    return x, W, b, y_


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x, W, b, y_ = initialization()
y = tf.nn.softmax(tf.matmul(x, W) + b)
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# 需要定义优化方法和优化目标，这里方法是梯度下降，目标使我们定义的交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 进行初始化参数，并且运行session
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# 每100条数据为1个batch，将数据feed给model
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
