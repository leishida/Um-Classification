import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import seaborn as sns
import os
import random
from keras.optimizers import SGD, Adam
from models import MultiLayerPerceptron

def lr_scheduler(dataset, sets, mode):
    if mode == 'SSC' or dataset == 'fashion' or dataset == 'kuzushiji':
        lr = 1e-5
    elif dataset == 'cifar10':
        lr = 2e-6
    return lr

def plot_curve(loss, nb_epoch, label, phase = 'test', dataset = 'MNIST'):
    sns.set()
    if phase == 'test':
        fig_num = 1
        ylim = (0, 0.4)
        ylabel = 'Classification error (%)'
    else:
        fig_num = 2
        ylim = (0, 5)
        ylabel = 'Training loss'
    plt.figure(num = fig_num)
    plt.xlim(0, nb_epoch, 1)
    plt.ylim(ylim)
    plt.plot(range(0, nb_epoch), loss, label=label)
    plt.title(dataset + ', ' + phase, size=16)
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.legend()
    plt.grid(True)
        
def get_Pi(sets, mode):
    # uniform random priors frow [0.1, 0.9]
    Pi = np.random.rand(sets) * 0.8 + 0.1
    return Pi

def get_set_sizes(sets, data_len, mode):
    set_size = data_len // sets
    set_sizes = np.ones(sets) * set_size
    return set_sizes

def save_data(args, U_sets, priors_corr, prior_test, np_loss_train, np_loss_test):
    #Build and Make dir
    output_dir = os.path.dirname(os.path.realpath(__file__)) + '/output_' +\
                 args.mode + '/' + args.dataset + '/' + str(args.sets) + '/'    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    #Save figures
    plot_curve(np_loss_test, args.epoch, label=args.mode, phase = 'test', dataset=args.dataset)
    plot_curve(np_loss_train, args.epoch, label=args.mode, phase = 'train', dataset=args.dataset)
    plt.figure(1)
    plt.savefig(output_dir + "test_curve.png",format='png')
    plt.figure(2)
    plt.savefig(output_dir + "train_curve.png",format='png')
    plt.close('all')
    
    #Save Loss-Epoch Vector
    np.save(output_dir + 'train_loss' + '.npy', np_loss_train)
    np.save(output_dir + 'test_loss' + '.npy', np_loss_test)
    
    #Save Other Data and Records
    data_dir = output_dir + 'data/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    np.save(data_dir + 'U_sets.npy', U_sets)
    np.save(data_dir + 'priors_corr.npy', priors_corr)
    np.save(data_dir + 'prior_test.npy', [prior_test])
    np.save(data_dir + 'Pi.npy', args.Pi)
    np.save(data_dir + 'set_sizes.npy', args.set_sizes)
    np.save(data_dir + 'sets.npy', [args.sets])
    fp = open(data_dir + "info",'w+')
    fp.write(str(args.sets) + "sets\n")
    fp.write("set sizes:" + str(args.set_sizes) + "\n")
    fp.write("Pi_i:" + str(args.Pi) + "\n")
    fp.write("Pi_D:" + str(prior_test) + "\n")
    fp.write("rho_i:" + str(priors_corr) + "\n")
    fp.write("learing rate:" + str(args.learningrate) + "\n")
    fp.write("lr decay:" + str(args.lr_decay) + "\n")
    fp.close()     