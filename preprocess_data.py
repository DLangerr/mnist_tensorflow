# -*- coding: utf-8 -*-
"""
Created on Tue May  8 16:48:19 2018

@author: Daniel
"""

import pandas as pd
import numpy as np
import os

def preprocess_data():

    data = pd.read_csv('MNIST_data.csv')
    data = data.as_matrix()

    X = data[:,1:]
    X[X>0] = 1
    X.astype(int)
    Y = data[:,0]
    N = len(X)
    K = 10

    Y_one_hot = one_hot_encode(Y, K)
    
    return X, Y, Y_one_hot


def one_hot_encode(Y, K):
    
    Y_one_hot = np.zeros([Y.shape[0], K])
 
    for i in range(Y.shape[0]):
        Y_one_hot[i, Y[i]] = 1
       
    return Y_one_hot
        

if __name__ == '__main__':
    preprocess_data()