#!/usr/bin/env python
#

import utils
import Training
import RBM

import copy
import sys
import numpy as np
import gzip
try:
    import cPickle as pickle
except:
    import pickle

                     # RBM parameters
command_line_args = {'epochs': (50, int),
                     'visibleUnits': (784, int),
                     'hiddenUnits': (500, int),
                     'approxMethod': ('CD', str),
                     'approxSteps': (1, int),
                     'persistStart': (100, int),
                     # Learning parameters
                     'batchSize': (100, int),
                     'learnRate': (0.005, float),
                     'momentum': (0.0, float),
                     'decayMagnitude': (0.0, float),
                     'decayType': ('l1', str),
                     'sigma': (0.001, float),
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
    epochs         = args['epochs']
    visibleUnits   = args['visibleUnits']
    hiddenUnits    = args['hiddenUnits']
    approxMethod   = args['approxMethod']
    approxSteps    = args['approxSteps']
    learnRate      = args['learnRate']
    persistStart   = args['persistStart']
    momentum       = args['momentum']
    decayMagnitude = args['decayMagnitude']
    decayType      = args['decayType']
    sigma          = args['sigma']
    batchSize      = args['batchSize']

    trace_file     = args['trace_file']  # saving results
    save_file      = args['save_file']  # saving parameters
    load_file      = args['load_file']  # loading parameters

    print('Loading data')
    with gzip.open('../data/mnist.pkl.gz', 'r') as f:
        # combine train and valid and leave test
        (TrainSet, y_train), (x_test, y_test), (ValidSet, y_Valid) = pickle.load(f, encoding='latin1')

    TrainSet = np.concatenate((TrainSet, x_test), axis=0)  # following authors 60000 pictures for training
    TrainSet = utils.binarize(TrainSet, threshold=0.001).T    # Create binary data
    ValidSet = utils.binarize(ValidSet, threshold=0.001).T    # Create binary data

    if len(load_file) == 0:
        print('Initializing model')
        model = RBM.RBM(n_vis=visibleUnits, n_hid=hiddenUnits,
                      momentum  = momentum,
                      sigma     = sigma,
                      TrainData = TrainSet,
                      wiseStart = True,
        )
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
        params = RBM.RBM.load(load_file)
        model = RBM.RBM(params=params)
        print(model.W.shape, model.vbias.shape, model.hbias.shape)


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
    if have_flag :
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
