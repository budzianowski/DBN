
'''
TODO:
1. doc strings
2.
3.

'''

import parsing
import theano as th
import theano.tensor as T
import numpy as np
import gzip
import time

try:
    import cPickle as pickle
except:
    import pickle
import sys
import copy
from theano import shared
from matplotlib.pyplot import plot

#   to add another command line argument, simply add:
#   its name as a key
#   value as a tuple of its default value and the argument type (e.g. int, string, float)
command_line_args = {'seed': (15485863, int),
                     'visibleSize': (2, int),
                     'hiddenSize': (3, int),
                     'learningRate': (0.01, float)}
# to add a new flag, simply add its name
command_line_flags = ['continuous']

def sigm(x):
    return (1 + np.exp(-x))**(-1)

class RBM(object):
    def __init__(self, visibleSize, hiddenSize, learningRate):

        # Initialization of class parameters
        self.visibleSize = visibleSize
        self.hiddenSize = hiddenSize
        self.sigmaInit = 0.01  # variance for initialization of parameters
        self.learningRate = learningRate

        # variables
        self.v = np.random.normal(0, self.sigmaInit, (1, visibleSize))  # convention from Krzakala's paper
        self.h = np.random.normal(0, self.sigmaInit, (hiddenSize, 1))

        # parameters
        self.W = np.random.normal(0, self.sigmaInit, (visibleSize, hiddenSize))
        self.a = np.random.normal(0, self.sigmaInit, (1, visibleSize))
        self.b = np.random.normal(0, self.sigmaInit, (hiddenSize, 1))
        self.params = [self.W, self.a, self.b]

        # magnetisations
        self.mV = np.random.normal(0, self.sigmaInit, (1, visibleSize))
        self.mH = np.random.normal(0, self.sigmaInit, (hiddenSize, 1))


    def iterate(self, N=10):  # pg.4
        for ii in range(N):
            # hidden magnetisations
            WmV = np.dot(self.mV - self.mV**2, self.W**2).T
            self.mH = sigm(self.b + np.dot(self.mV, self.W).T - np.multiply(self.mH - .5, WmV))
            # visible magnetisations
            WmH = np.dot(self.W**2, self.mH - self.mH**2).T
            self.mV = sigm(self.a + np.dot(self.W, self.mH).T - np.multiply(self.mV - .5, WmH))

    def getLL(self):

        pass

    def getGradient(self):
        gradW = np.zeros(self.W.shape)
        # first term
        for ii in range(self.W.shape[0]):
            for jj in range(self.W.shape[1]):
                gradW[ii, jj] = self.mV[0][ii] * self.mH[jj]
        # second term
        for ii in range(self.visibleSize):
            for jj in range(self.hiddenSize):
                gradW[ii, jj] += gradW[ii, jj] * (self.mV[0][ii] - self.mV[0][ii]**2) * (self.mH[jj] - self.mH[jj]**2)
        # TODO: third and fourth term

        gradA = -self.mV
        gradB = -self.mH

        gradients = [gradW, gradA, gradB]
        return gradients


    def update(self, gradients):
        # TODO: add regularization MAP? linear?
        # TODO: check if it's updating
        updates = []
        for param, grad in zip(self.params, gradients):
            updates.append(param + self.learningRate * grad)
        return updates

    def train(self): # this in the loop
        # iterate

        # getGradient
        gradients = self.getGradient()
        # update
        self.params = self.update(gradients)

        pass
if __name__ == "__main__":
    args = parsing.parse_args(command_line_args, command_line_flags)
    parsing.print_args(args)
    # model specification
    np.random.seed(args['seed'])
    visibleSize = args['visibleSize']
    hiddenSize = args['hiddenSize']
    learningRate = args['learningRate']

    continuous = args['continuous']

    model = RBM(visibleSize, hiddenSize, learningRate)
    print(model.W)
    print(sigm(np.array([-2,2])))