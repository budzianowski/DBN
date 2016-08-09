#!/usr/bin/env python
#
"""This module provides an interface for running
basic experiments on MNIST dataset.
"""

import Training
import DBN

import copy
import sys
import numpy as np
import gzip
import sklearn
try:
    import cPickle as pickle
except:
    import pickle

                     # DBN parameters
layers = [784, 500, 250, 25]
                     # RBM parameters
command_line_args = {'whatLayer': (2, int),
                     'epochs': (50, int),
                     'visibleUnits': (784, int),
                     'hiddenUnits': (500, int),
                     'secondLayerUnits': (250, int),
                     'approxMethod': ('CD', str),  # CD, naive, tap2 or tap3
                     'approxSteps': (1, int),
                     'persistStart': (0, int),
                     # Learning parameters
                     'batchSize': (100, int),
                     'learnRate': (0.01, float),
                     'momentum': (0.0, float),
                     'decayMagnitude': (0.00, float),
                     'decayType': ('l1', str),
                     'sigma': (0.001, float),
                     'binarization': (0.001, float),
                     # I/O settings
                     'trace_file': ('', str),          #if set, trace information will be written about number of training
                     'save_file': ('', str),           #if set will train model and save it to this file
                     'load_file': ('', str),
                     }
command_line_flags = ['continuous']


def main():
    # Set parameters and print them
    args = parse_args()
    print_args(args)
    whatLayer      = args['whatLayer']
    epochs         = args['epochs']
    visibleUnits   = args['visibleUnits']
    hiddenUnits    = args['hiddenUnits']
    secondLayerUnits = args['secondLayerUnits']
    approxMethod   = args['approxMethod']
    approxSteps    = args['approxSteps']
    learnRate      = args['learnRate']
    persistStart   = args['persistStart']
    momentum       = args['momentum']
    decayMagnitude = args['decayMagnitude']
    decayType      = args['decayType']
    sigma          = args['sigma']
    batchSize      = args['batchSize']
    binarization   = args['binarization']

    trace_file     = args['trace_file']  # saving results
    save_file      = args['save_file']  # saving parameters
    load_file      = args['load_file']  # loading parameters

    print('Loading data')
    with gzip.open('../data/mnist.pkl.gz', 'r') as f:
        # combine train and valid and leave test
        (TrainSet, y_train), (x_test, y_test), (ValidSet, y_Valid) = pickle.load(f, encoding='latin1')

    TrainSet = np.concatenate((TrainSet, x_test), axis=0)
    TrainSet = sklearn.preprocessing.binarize(TrainSet, threshold=binarization).T  # Create binary data
    ValidSet = sklearn.preprocessing.binarize(ValidSet, threshold=binarization).T  # Create binary data

    if len(load_file) == 0:
        for ii in range(len(layers) - 1):
            if ii == 0:
                print('Initializing model')
                model = DBN.DBN(n_vis=layers[ii], n_hid=layers[ii + 1],
                                momentum  = momentum,
                                sigma     = sigma,
                                trainData = TrainSet,
                                wiseStart = True,
                                )

            else:
                params = DBN.DBN.load("DBN")
                model = DBN.DBN(layer=ii+1, params=params, n_vis=layers[ii], n_hid=layers[ii + 1])

            print('Start of training')
            Training.fit(model, TrainSet, ValidSet,
                         n_epochs         = epochs,
                         weight_decay     = decayType,
                         decay_magnitude  = decayMagnitude,
                         lr               = learnRate,
                         batch_size       = batchSize,
                         NormalizationApproxIter = approxSteps,
                         approx           = approxMethod,
                         persistent_start = persistStart,
                         trace_file       = trace_file,
                         save_file        = save_file
            )

    else:
        # place for plotting with learned structure
        print('Initializing model')
        params = DBN.DBN.load(load_file)
        model = DBN.DBN(layer=whatLayer, params=params, n_vis=hiddenUnits, n_hid=secondLayerUnits)
        print('W1 shape', model.W1.shape, 'hbias', model.hbias1.shape, 'vbias', model.vbias1.shape)
        if model.layer == 3 or model.layer == 4:
            print('W2 shape', model.W22.shape, 'hbias', model.hbias2.shape, 'vbias', model.vbias2.shape)
        if model.layer == 4:
            print('W3 shape', model.W33.shape, 'hbias', model.hbias3.shape, 'vbias', model.vbias3.shape)
        print('Next W shape', model.W.shape, 'hbias', model.hbias.shape, 'vbias', model.vbias.shape)
        print('Start of training')


def get_arg(arg, args, default, type_):
    arg = '--'+arg
    if arg in args:
        index = args.index(arg)
        value = args[args.index(arg) + 1]
        del args[index]     #remove arg-name
        del args[index]     #remove value
        return type_(value)
    else:
        return default


def get_flag(flag, args):
    flag = '--'+flag
    have_flag = flag in args
    if have_flag:
        args.remove(flag)

    return have_flag


def parse_args():
    args = copy.deepcopy(sys.argv[1:])
    arg_dict = {}
    for (arg_name, arg_args) in command_line_args.items():
        (arg_defalut_val, arg_type) = arg_args
        arg_dict[arg_name] = get_arg(arg_name, args, arg_defalut_val, arg_type)

    for flag_name in command_line_flags:
        arg_dict[flag_name] = get_flag(flag_name, args)

    if len(args) > 0:
        print('Have unused args: {0}'.format(args))

    return arg_dict


def print_args(args):
    print('Parameters used:')
    print('--------------------------------------')
    for (k, v) in args.items():
        print('\t{0}: {1}'.format(k, v))
    print('--------------------------------------')


if __name__ == '__main__':
    main()
