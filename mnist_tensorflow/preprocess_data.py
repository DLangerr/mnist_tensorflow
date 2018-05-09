# -*- coding: utf-8 -*-
"""
Created on Tue May  8 16:48:19 2018

@author: Daniel
"""

import pandas as pd
import numpy as np

def preprocess_data():
    data = pd.read_csv('MNIST_data.csv')
    data = data.as_matrix()
    X = data[:,1:]
    Y = data[:,0]
    D = X.shape[1]
    N = len(X)
    K = len(set(Y))
    
    Y_one_hot = one_hot_encode(Y, K)
    
    return X, Y, Y_one_hot, D, N, K


def one_hot_encode(Y, K):
    
    Y_one_hot = np.zeros([Y.shape[0], K])
 
    for i in range(Y.shape[0]):
        Y_one_hot[i, Y[i]] = 1
       
    return Y_one_hot
        

if __name__ == '__main__':
    preprocess_data()