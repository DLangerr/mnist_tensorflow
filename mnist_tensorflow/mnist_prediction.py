# -*- coding: utf-8 -*-
from preprocess_data import preprocess_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def forward(X, W1, b1, W2, b2, W3, b3):
    Z1 = tf.nn.tanh(tf.matmul(X, W1) + b1)
    Z2 = tf.nn.tanh(tf.matmul(Z1, W2) + b2)
    return tf.matmul(Z2, W3) + b3

def error_rate(Y_, Y):
    return np.mean(np.equal(tf.argmax(Y_, 1), Y).astype(float))

def train_test_split(Z, train_percentage=0.7):
   
    Z_train_size = int(Z.shape[0]*train_percentage)
    Z_test_size = Z.shape[0]-Z_train_size
    Z_train = Z[:Z_train_size]
    Z_test = Z[:Z_test_size]
    
    return Z_train, Z_test

X, Y, Y_one_hot, D, N, K = preprocess_data()

X_train, X_test = train_test_split(X)
Y_train, Y_test = train_test_split(Y)
Y_train_ohe, Y_test_ohe = train_test_split(Y_one_hot)

phX = tf.placeholder(tf.float32, [None, D])
phY = tf.placeholder(tf.float32, [None, K])

M1 = 20
M2 = 10
lr = 0.01


W1 = tf.Variable(tf.random_normal([D, M1], stddev=0.1))
b1 = tf.Variable(tf.zeros([M1]))
W2 = tf.Variable(tf.random_normal([M1, M2], stddev=0.1))
b2 = tf.Variable(tf.zeros([M2]))
W3 = tf.Variable(tf.random_normal([M2, K], stddev=0.1))
b3 = tf.Variable(tf.zeros([K]))

Y_ = forward(phX, W1, b1, W2, b2, W3, b3)
pred = tf.argmax(Y_, 1)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=phY, logits=Y_))
train = tf.train.AdamOptimizer(lr).minimize(cost)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

init = tf.global_variables_initializer()

sess.run(init)

epochs = 100
cost_array = []

for i in range(epochs):
    sess.run(train, feed_dict={phX:X_train, phY:Y_train_ohe})
    c = np.mean(sess.run(cost, feed_dict={phX:X_train, phY:Y_train_ohe}))
    
    if i % 1 == 0:
        predictions = sess.run(pred, feed_dict={phX:X_test})
        acc = np.mean(predictions==Y_test)
        print(f"Iteration {i}. Cost: {c}. Accuracy: {acc}")
    
    cost_array.append(c)

plt.plot(cost_array)
plt.show()





