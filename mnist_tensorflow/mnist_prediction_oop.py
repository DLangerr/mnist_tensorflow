# -*- coding: utf-8 -*-
from preprocess_data import preprocess_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class NN(object):
    def __init__(self, D, M, K, N, X, Y, Y_one_hot):
       
        self.M = M
        self.D = D
        self.K = K
        self.X = X
        self.Y = Y
        self.Y_one_hot = Y_one_hot
        
        self.X_train, self.X_test = self.train_test_split(self.X)
        self.Y_train, self.Y_test = self.train_test_split(self.Y)
        self.Y_train_ohe, self.Y_test_ohe = self.train_test_split(self.Y_one_hot)

        self.N = N
        self.build(D, M, K)
        
    def build(self, D, M, K):
        self.W1 = tf.Variable(tf.random_normal([D, M], stddev=0.1))
        self.b1 = tf.Variable(tf.zeros([M]))
        self.W2 = tf.Variable(tf.random_normal([M, K], stddev=0.1))
        self.b2 = tf.Variable(tf.zeros([K]))

        
        
        self.phX = tf.placeholder(tf.float32, [None, D])
        self.phY = tf.placeholder(tf.float32, [None, K])

        self.Y_ = self.forward(self.phX)
        self.pred = tf.argmax(self.Y_, 1)
        
        

    def forward(self, X):
        self.Z1 = tf.nn.tanh(tf.matmul(X, self.W1) + self.b1)
        return tf.matmul(self.Z1, self.W2) + self.b2


    
    
    def fit(self, epochs, lr=0.01):
          
        
        
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.phY, logits=self.Y_))
        self.train = tf.train.AdamOptimizer(lr).minimize(self.cost)
        
        cost_array = []
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            for i in range(epochs):
                sess.run(self.train, feed_dict={self.phX:self.X_train, self.phY:self.Y_train_ohe})
                c = np.mean(sess.run(self.cost, feed_dict={self.phX:self.X_train, self.phY:self.Y_train_ohe}))
    
              
                predictions = sess.run(self.pred, feed_dict={self.phX:self.X_test})
                acc = np.mean(predictions==self.Y_test)
                print(f"Iteration {i}. Cost: {c}. Accuracy: {acc}")
    
                cost_array.append(c)

        plt.plot(cost_array)
        plt.show()


    def train_test_split(self, Z, train_percentage=0.7):
   
        Z_train_size = int(Z.shape[0]*train_percentage)
        Z_test_size = Z.shape[0]-Z_train_size
        Z_train = Z[:Z_train_size]
        Z_test = Z[:Z_test_size]
    
        return Z_train, Z_test

def main():
    X, Y, Y_one_hot, D, N, K = preprocess_data()
    
    nn = NN(D, 50, K, N, X, Y, Y_one_hot)
    nn.fit(100)
    
    
    


if __name__ == '__main__':
    main()









