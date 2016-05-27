#!/usr/bin/env python
#

import utils
import Training2
import RBMBase
import numpy as np
import gzip
try:
    import cPickle as pickle
except:
    import pickle


def run_mnist():
    # Set parameters
    Epochs         = 20
    VisibleUnits   = 784
    HiddenUnits    = 500
    Approx         = ["tap2"]
    ApproxSteps    = 10
    LearnRate      = 0.005
    PersistStart   = 0
    Momentum       = 0
    DecayMagnitude = 0.01
    DecayType      = "l1"
    Sigma          = 0.001
    BatchSize      = 100

    print('loading data')
    with gzip.open('mnist.pkl.gz', 'r') as f:
        # combine train and valid and leave test
        (TrainSet, y_train), (x_test, y_test), (ValidSet, y_Valid) = pickle.load(f, encoding='latin1')

    TrainSet = np.concatenate((TrainSet, x_test), axis=0)  # following authors 60000 pictures for training

    TrainSet = utils.binarize(TrainSet, threshold=0.001).T    # Create binary data
    ValidSet = utils.binarize(ValidSet, threshold=0.001).T    # Create binary data

    print('initializing model')
    rbm = RBMBase.RBM(VisibleUnits, HiddenUnits,
                      momentum  = Momentum,
                      sigma     = Sigma,
                      TrainData = TrainSet,
                      wiseStart = True,
                      batchSize = BatchSize
    )

    print('start of training')
    Training2.fit(rbm, TrainSet, ValidSet,
                 n_epochs         = Epochs,
                 weight_decay     = DecayType,
                 decay_magnitude  = DecayMagnitude,
                 lr               = LearnRate,
                 NormalizationApproxIter = ApproxSteps,
                 approx           = Approx,
                 persistent_start = PersistStart
    )

if __name__ == '__main__':
    run_mnist()
