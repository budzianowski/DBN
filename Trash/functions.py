import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def derivativeSigmoid(z):
    return sigmoid(z)*(1.0 - sigmoid(z))
