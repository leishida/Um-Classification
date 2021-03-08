from __future__ import print_function
import sys
import os
import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from helper import plot_curve, save_data, lr_scheduler, get_Pi, get_set_sizes
from models import MultiLayerPerceptron
from dataset import load_dataset, get_U_sets
import argparse
import random
from keras.optimizers import SGD, Adam

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

def get_args():
    parser = argparse.ArgumentParser(
        description='UU learning Keras implementation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--sets', type=int, default=20)
    parser.add_argument('--sets_sizes')
    parser.add_argument('--Pi')
    
    parser.add_argument('--Pi_gen', type=str, default="random", choices=['random'])
    parser.add_argument('--set_size_gen', type=str, default="uniform", choices=['uniform'])
    parser.add_argument('--eps',type=float,default=0)
    parser.add_argument('--data_gen', type=str, default="random", choices=['random'])
    
    parser.add_argument('--batchsize', type=int, default=512)
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist'])
    parser.add_argument('--mode', type=str, default='SSC', choices=['SSC'])
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp'])
    parser.add_argument('--learningrate', type=float, default=-1)
    parser.add_argument('--lr_decay', type=float, default=1e-4)
    parser.add_argument('--weightdecay', type=float, default=1e-4)
    parser.add_argument('--optimizer',type=str,default='Adam', choices=['Adam', 'SGD'])
    args = parser.parse_args()
    return args

def exp(args): 
    #Get Image Data
    x_train, y_train, x_test, y_test, prior_test = load_dataset(args.dataset)
    
    #Get Sets Sizes
    args.set_sizes = get_set_sizes(args.sets, len(x_train), args.set_size_gen)
    
    #Randomly Generate Priors Pi
    args.Pi = get_Pi(args.sets, args.Pi_gen)
    
    #Sample Data According to Pi and Sets Sizes
    U_sets, priors_corr = get_U_sets(args.sets, y_train, args.set_sizes, args.Pi)
    
    print('Data prepared!')
    print("set_sizes: " + str(args.set_sizes))
    print("test class prior: " + str(prior_test))
    print("Pi: " + str(args.Pi))

    # Get Model
    ExpModel = MultiLayerPerceptron(dataset=args.dataset,
                    sets = args.sets,
                    set_sizes = args.set_sizes,
                    Pi = args.Pi,
                    mode=args.mode,
                    weight_decay=args.weightdecay)
    
    # Schedule Learning Rate if not specified
    if args.learningrate == -1:
        args.learningrate = lr_scheduler(args.dataset, args.sets, args.mode)
        
    # Get optimizer
    ExpModel.optimizer = Adam(args.optimizer, lr=args.learningrate, decay=args.lr_decay)
    
    # Build Model
    input_shape = x_train[0].shape
    ExpModel.build_model(priors_corr, prior_test, args.Pi, input_shape, mode = args.mode)

#-----------------------------------------------------Start Training-----------------------------------------------------#
    history, loss_test = ExpModel.fit_model(U_sets=U_sets, x_train_total = x_train,
                                            batch_size=args.batchsize,
                                            epochs=args.epoch,
                                            x_test=x_test,
                                            y_test=y_test,
                                            Pi=args.Pi,
                                            priors_corr = priors_corr,
                                            prior_test = prior_test,
                                            mode = args.mode
                                            )
    np_loss_test = np.array(loss_test)
    np_loss_train = np.array(history['loss'])

    plot_curve(np_loss_test, args.epoch, label=args.mode, phase = 'test', dataset=args.dataset)
    plot_curve(np_loss_train, args.epoch, label=args.mode, phase = 'train', dataset=args.dataset)

#---------------------------------------------Save files----------------------------------------------------------------#
    save_data(args, U_sets, priors_corr, prior_test, np_loss_train, np_loss_test)
                                                                    
if __name__ == '__main__':
    args = get_args()
    print("mode: {}".format(args.mode))
    print("model: {}".format(args.model))
    print("sets: " + str(args.sets))
    
    exp(args)
