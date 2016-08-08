from toyBias import *
import numpy as np
import copy

# definitions
n = 8  # size of layer
lattice = LatticeRBM(n)
print('truth', 'sequentialMF', 'defaultMF', 'sequentialTAP', 'defaultTAP')
for itr in range(10):
    W = np.random.random((n, n)) - .5
    #W = np.ones((n, n)) / -2.
    F = -np.log(magnetizationRBM(lattice, W, n))
    magReal = copy.deepcopy(lattice.m)

    lattice.m = np.ones((2*n, 1)) / 2.0
    for ii in range(10):
        updateRBM1(lattice, W, n, "naive")
    sequential = partitionRBM1(lattice, W, n, "tap")

    lattice.m = np.ones((2*n, 1)) / 2.0
    for ii in range(10):
        updateRBM2(lattice, W, n, "naive")
    default = partitionRBM2(lattice, W, n, "tap")
    magMF = np.concatenate((lattice.hid, lattice.vis), axis=0)

    lattice.m = np.ones((2*n, 1)) / 2.0
    for ii in range(10):
        updateRBM1(lattice, W, n, "tap")
    sequentialTap = partitionRBM1(lattice, W, n, "tap")

    lattice.m = np.ones((2*n, 1)) / 2.0
    for ii in range(10):
        updateRBM2(lattice, W, n, "tap")
    defaultTap = partitionRBM2(lattice, W, n, "tap")
    magTAP = np.concatenate((lattice.hid, lattice.vis), axis=0)

    print(magMSE(magReal, magMF), magMSE(magReal, magTAP))
    print(F, sequential, default, sequentialTap, defaultTap)

