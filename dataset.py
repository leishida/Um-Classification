from __future__ import print_function
import numpy as np
import scipy.io
import random
import keras, wget
from keras.datasets import mnist
import os

def get_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255
    return (x_train, y_train), (x_test, y_test)

def binarize_mnist_class(_trainY, _testY):
    trainY = np.ones(len(_trainY), dtype=np.int32).reshape(len(_trainY), 1)
    trainY[_trainY % 2 == 1] = 0
    testY = np.ones(len(_testY), dtype=np.int32).reshape(len(_testY), 1)
    testY[_testY % 2 == 1] = 0
    return trainY, testY

def load_dataset(dataset_name):
    (trainX, trainY), (testX, testY) = get_mnist()
    trainY, testY = binarize_mnist_class(trainY, testY)
    prior_test = (testY == 1).sum() / float(len(testY))
    return trainX, trainY, testX, testY, prior_test

def get_U_sets(bags, y_train, bag_sizes, thetas):
    pos_idx = [i for i, x in enumerate(y_train) if x == 1]
    neg_idx = [i for i, x in enumerate(y_train) if x == 0]
    U_sets = ()
    total_size = sum(bag_sizes)
    priors_corr = [bag_size / sum(bag_sizes) for bag_size in bag_sizes]
    for i in range(bags):
        np.random.shuffle(pos_idx)
        np.random.shuffle(neg_idx)
        n_pos = int(bag_sizes[i] * thetas[i])
        n_neg = int(bag_sizes[i] - n_pos)
        cur_set = np.concatenate((pos_idx[:n_pos], neg_idx[:n_neg])).astype(int)
        np.random.shuffle(cur_set)
        U_sets = U_sets + (cur_set,)
    
    return U_sets, priors_corr