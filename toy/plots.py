from toy import *
import numpy as np

import numpy as np
from numpy import genfromtxt
import random
import copy

##import matplotlib.pyplot as plt
ww = np.arange(-1.0, 1.0, step=.025)
'''
# Constant weight from -1 to 1
n = 4  # size of grid
lattice = [[Node() for x in range(n)] for y in range(n)]

results = np.zeros((0, 6))
for ii in ww:
    coupl = ii
    W = [[[coupl, coupl, coupl, coupl] for x in range(n)] for y in range(n)]
    for iter in range(20):
        updateMF(lattice, W, n)
    mf = partitionMF(lattice, W, n)
    mfbytap = partition(lattice, W, n, "tap2")
    magMF = getMag(lattice, n)
    reset(lattice, W, n)
    for iter in range(20):
        updateTAP2(lattice, W, n)
    tap = partitionTAP(lattice, W, n)
    tapbymf = partition(lattice, W, n, "naive")
    magTAP = getMag(lattice, n)
    reset(lattice, W, n)
    Z = magnetization(lattice, W, n)
    magReal = getMag(lattice, n)
    # results
    results = np.vstack((results, [-np.log(Z), mf, mfbytap, tap, magMSE(magReal, magMF), magMSE(magReal, magTAP)]))
np.save('saves', results)

results = np.load('saves.npy')
fig, ax1 = plt.subplots(1)
ax1.plot(ww, results[:, 0], 'g',  label=r'$-\log(Z)$', linewidth=8.0)
ax1.plot(ww, results[:, 1], 'bv', markersize=9, label=r'MF')
#ax1.plot(ww, results[:, 2], 'k.', markersize=10, label=r'MFbyTAP')
ax1.plot(ww, results[:, 3], 'r^', markersize=9, label=r'TAP')
ax1.legend(loc=1)
ax1.set_ylabel('free energy')
ax1.set_xlabel(r'couplings value')
fig.savefig('sameCouplingsZ.pdf')

fig, ax2 = plt.subplots(1)
ax2.plot(ww, results[:, 4], 'b', linewidth=6.0, label=r'MF magnetizations')
ax2.plot(ww, results[:, 5], 'r', linewidth=6.0, label=r'TAP magnetizations')
ax2.legend(loc='best')
ax2.set_ylabel('MSE')
ax2.set_xlabel(r'couplings value')
fig.savefig('sameCouplingsMAG.pdf')
plt.show()


## change to absolute power
n = 4  # size of grid
lattice = [[Node() for x in range(n)] for y in range(n)]
random.choice([1, -1])
ww = np.arange(.1, 1, step=.1)
results = np.zeros((10, 9, 5))
for itr in range(10):
    print(itr)
    res = np.zeros((0, 5))
    for ii in ww:
        W = [[[.2 * np.random.rand() -.1 + random.choice([1, -1])*ii, .2 * np.random.rand() -.1 + random.choice([1, -1])*ii, .2 * np.random.rand() -.1 + random.choice([1, -1])*ii, .2 * np.random.rand() -.1 + random.choice([1, -1])*ii] for x in range(n)] for y in range(n)]
        for iter in range(20):
            updateMF(lattice, W, n)
        mf = partitionMF(lattice, W, n)
        mfbytap = partition(lattice, W, n, "tap2")
        magMF = getMag(lattice, n)
        reset(lattice, W, n)
        for iter in range(20):
            updateTAP2(lattice, W, n)
        tap = partitionTAP(lattice, W, n)
        tapbymf = partition(lattice, W, n, "naive")
        magTAP = getMag(lattice, n)
        reset(lattice, W, n)
        Z = magnetization(lattice, W, n)
        magReal = getMag(lattice, n)
        # results
        res = np.vstack((res, [abs(-np.log(Z) - mf), abs(-np.log(Z) - mfbytap), abs(-np.log(Z) -tap), magMSE(magReal, magMF), magMSE(magReal, magTAP)]))
    results[itr] = res

np.save('saves', results)

import matplotlib.pyplot as plt
ww = np.arange(.1, 1, step=.1)
results = np.load('saves.npy')
# mf - Z, mftap - Z, tap - Z,
print(results[9, 0, :])
means = np.zeros((9, 5))
sd = np.zeros((9, 5))
for ii in range(9):
    means[ii, :] = np.mean(results[:, ii, :], axis = 0)
    sd[ii, :] = np.std(results[:, ii, :], axis = 0)

fig, ax1 = plt.subplots(1)
ax1.errorbar(ww, means[:, 0], yerr=sd[:, 0], fmt='bv',ms=2,  elinewidth=2, label=r'MF')
#ax1.errorbar(ww, means[:, 1], yerr=sd[:, 1], fmt='k', label=r'MFbyTAP')
ax1.errorbar(ww+ 0.005, means[:, 2], yerr=sd[:, 2], fmt='rv', ms=2, elinewidth=2, label=r'TAP')
ax1.legend(loc=2)
ax1.set_ylabel('abs. difference from true free energy')
ax1.set_xlabel(r'absolute couplings strength')
plt.xlim(0.05, 0.95)
#plt.ylim(-0.01, 1)
plt.show()
fig.savefig('randomWeightsZ.pdf')

fig, ax2 = plt.subplots(1)
ax2.errorbar(ww, means[:,3], yerr=sd[:, 3], fmt='bo', elinewidth=2, label=r'MF magnetizations')
ax2.errorbar(ww + 0.005, means[:,4], yerr=sd[:, 4], fmt='ro', elinewidth=2,  label=r'TAP magnetizations')
ax2.set_ylabel('MSE')
ax2.set_xlabel(r'absolute couplings strength')
ax2.legend(loc=2)
plt.xlim(0.05, 0.95)
plt.show()
fig.savefig('randomWeightsMAG.pdf')

'''
##########################
# RBM
## Constant weight from -1 to 1
'''
n = 10 # size of layer
lattice = LatticeRBM(n)

ww = np.arange(-1.0, 1.0, step=.025)
results = np.zeros((0, 11))
for ii in ww:
    print(ii)
    coupl = ii
    W = np.ones((n, n)) * coupl

    lattice.m = np.ones((2*n, 1)) / 2.0
    for ii in range(10):
        updateRBM1(lattice, W, n, "naive")
    sequential = partitionRBM1(lattice, W, n, "naive")

    lattice.m = np.ones((2*n, 1)) / 2.0
    for ii in range(10):
        updateRBM2(lattice, W, n, "naive")
    default = partitionRBM2(lattice, W, n, "naive")
    magMF = np.concatenate((lattice.hid, lattice.vis), axis=0)

    lattice.m = np.ones((2*n, 1)) / 2.0
    lattice.vis = np.ones((n, 1)) / 2.0; lattice.hid = np.ones((n, 1)) / 2.0
    for ii in range(10):
        updateRBM3(lattice, W, n, "naive")
    parallel = partitionRBM2(lattice, W, n, "naive")
    parMF = np.concatenate((lattice.hid, lattice.vis), axis=0)

    lattice.m = np.ones((2*n, 1)) / 2.0
    lattice.vis = np.ones((n, 1)) / 2.0; lattice.hid = np.ones((n, 1)) / 2.0
    for ii in range(10):
        updateRBM3(lattice, W, n, "tap")
    parallelTAP = partitionRBM2(lattice, W, n, "tap")
    parTAP = np.concatenate((lattice.hid, lattice.vis), axis=0)

    lattice.m = np.ones((2*n, 1)) / 2.0
    for ii in range(10):
        updateRBM1(lattice, W, n, "tap")
    sequentialTAP = partitionRBM1(lattice, W, n, "tap")
    lattice.m = np.ones((2*n, 1)) / 2.0
    lattice.vis = np.ones((n, 1)) / 2.0; lattice.hid = np.ones((n, 1)) / 2.0
    for ii in range(10):
        updateRBM2(lattice, W, n, "tap")
    defaultTap = partitionRBM2(lattice, W, n, "tap")
    magTAP = np.concatenate((lattice.hid, lattice.vis), axis=0)

    F = -np.log(magnetizationRBM(lattice, W, n))
    magReal = copy.deepcopy(lattice.m)

    # results
    results = np.vstack((results, [F, sequential, default, sequentialTAP, defaultTap, magMSE(magReal, magMF), magMSE(magReal, magTAP), \
                                   parallel, parallelTAP, magMSE(magReal, parMF), magMSE(magReal, parTAP)]))
np.save('saves', results)


import matplotlib.pyplot as plt
results = np.load('saves.npy')
fig, ax1 = plt.subplots(1)
ax1.plot(ww, results[:, 0], 'g',  linewidth=10.0, label=r'$-\log(Z)$')
#ax1.plot(ww, results[:, 1], 'rv', label=r'sequentialMF')
#ax1.plot(ww, results[:, 3], 'gv', label=r'sequentialTAP')
ax1.plot(ww, results[:, 2], 'bv',markersize=6, label=r'asynchronous MF')
ax1.plot(ww, results[:, 4], 'rv',markersize=6, label=r'asynchronous TAP')
ax1.plot(ww, results[:, 7], 'bo',markersize=6, label=r'parallel MF')
ax1.plot(ww, results[:, 8], 'ro',markersize=6, label=r'parallel TAP')

ax1.legend(loc=3)
ax1.set_ylabel('partition function')
ax1.set_xlabel(r'couplings value')
fig.savefig('sameCouplingsRBMZ.pdf')
plt.show()

fig, ax2 = plt.subplots(1)
ax2.plot(ww, results[:, 5], 'b^', markersize=6, label=r'asynchronous MF magnetizations')
ax2.plot(ww, results[:, 6], 'r^',markersize=6, label=r'asynchronous TAP magnetizations.')
ax2.plot(ww+0.01, results[:, 9], 'bv', markersize=6,label=r'parallel MF magnetizations.')
ax2.plot(ww + 0.01, results[:, 10], 'rv', markersize=6,label=r'parallel TAP magnetizations.')
ax2.set_ylabel('MSE')
ax2.legend(loc=1)
ax1.set_xlabel(r'couplings value')
fig.savefig('sameCouplingsRBMMAG.pdf')
plt.xlim(-0.98, 1)
plt.show()
'''

##  random values
'''
n = 10  # size of layer
lattice = LatticeRBM(n)
ww = np.arange(.1, 1, step=.1)
results = np.zeros((10, 9, 10))
for itr in range(10):
    print(itr)
    res = np.zeros((0, 10))
    for weight in ww:
        W = .2 * np.random.random((n, n)) -.1  # definition
        for ii in range(n):
            for jj in range(n):
                W[ii, jj] += random.choice([1, -1]) * weight

        lattice.m = np.ones((2*n, 1)) / 2.0
        lattice.vis = np.ones((n, 1)) / 2.0; lattice.hid = np.ones((n, 1)) / 2.0
        for ii in range(20):
            updateRBM2(lattice, W, n, "naive")
        default = partitionRBM2(lattice, W, n, "naive")
        magMF = np.concatenate((lattice.hid, lattice.vis), axis=0)

        lattice.m = np.ones((2*n, 1)) / 2.0
        lattice.vis = np.ones((n, 1)) / 2.0; lattice.hid = np.ones((n, 1)) / 2.0
        for ii in range(20):
            updateRBM1(lattice, W, n, "naive")
        sequential = partitionRBM1(lattice, W, n, "naive")

        lattice.m = np.ones((2*n, 1)) / 2.0
        lattice.vis = np.ones((n, 1)) / 2.0; lattice.hid = np.ones((n, 1)) / 2.0
        for ii in range(20):
            updateRBM2(lattice, W, n, "tap")
        defaultTAP = partitionRBM2(lattice, W, n, "tap")
        magTAP = np.concatenate((lattice.hid, lattice.vis), axis=0)

        lattice.m = np.ones((2*n, 1)) / 2.0
        lattice.vis = np.ones((n, 1)) / 2.0; lattice.hid = np.ones((n, 1)) / 2.0
        for ii in range(20):
            updateRBM1(lattice, W, n, "naive")
        sequentialTAP = partitionRBM1(lattice, W, n, "tap")

        lattice.m = np.ones((2*n, 1)) / 2.0
        lattice.vis = np.ones((n, 1)) / 2.0; lattice.hid = np.ones((n, 1)) / 2.0
        for ii in range(20):
            updateRBM3(lattice, W, n, "naive")
        parallel = partitionRBM2(lattice, W, n, "naive")
        parMF = np.concatenate((lattice.hid, lattice.vis), axis=0)

        lattice.m = np.ones((2*n, 1)) / 2.0
        lattice.vis = np.ones((n, 1)) / 2.0; lattice.hid = np.ones((n, 1)) / 2.0
        for ii in range(20):
            updateRBM3(lattice, W, n, "tap")
        parallelTAP = partitionRBM2(lattice, W, n, "tap")
        parTAP = np.concatenate((lattice.hid, lattice.vis), axis=0)

        F = -np.log(magnetizationRBM(lattice, W, n))
        magReal = copy.deepcopy(lattice.m)
        # results

        res = np.vstack((res, [abs(F - default)[0][0], abs(F - sequential)[0][0], abs(F - sequentialTAP)[0][0], abs(F - defaultTAP)[0][0], \
                               abs(F - parallel)[0][0], abs(F - parallelTAP)[0][0],\
                               magMSE(magReal, magMF)[0], magMSE(magReal, magTAP)[0], magMSE(magReal, parMF)[0], magMSE(magReal, parTAP)[0]]))
    results[itr] = res
np.save('saves', results)

import matplotlib.pyplot as plt
results = np.load('saves.npy')
ww = np.arange(.1, 1, step=.1)
# mf - Z, mftap - Z, tap - Z,
print(results[9, 0, :])
means = np.zeros((9, 10))
sd = np.zeros((9, 10))
for ii in range(9):
    means[ii, :] = np.mean(results[:, ii, :], axis = 0)
    sd[ii, :] = np.std(results[:, ii, :], axis = 0)

fig, ax1 = plt.subplots(1)
ax1.errorbar(ww, means[:, 0], yerr=sd[:, 0], fmt='bv', label=r'asynchronous MF')
ax1.errorbar(ww+0.01, means[:, 3], yerr=sd[:, 3], fmt='rv', label=r'asynchronous TAP')
#ax1.errorbar(ww+0.005, means[:, 1], yerr=sd[:, 1], fmt='b^', label=r'sequential MF')
#ax1.errorbar(ww, means[:, 2], yerr=sd[:, 2], fmt='r^', label=r'sequential TAP')
ax1.errorbar(ww+0.01, means[:, 4], yerr=sd[:, 4], fmt='bo', label=r'parallel MF')
ax1.errorbar(ww+0.005, means[:, 5], yerr=sd[:, 5], fmt='ro', label=r'parallel TAP')
ax1.legend(loc=2)
ax1.set_ylim([0, 1])
ax1.set_ylabel('abs. difference from true free energy')
ax1.set_xlabel(r'absolute couplings strength')
plt.show()
fig.savefig('randomWeightsRBMZ.pdf')

fig, ax2 = plt.subplots(1)
ax2.errorbar(ww, means[:,6], yerr=sd[:, 6], fmt='bo', label=r'asynchronous MF magn.')
ax2.errorbar(ww+0.005, means[:,7], yerr=sd[:, 7], fmt='ro', label=r'asynchronous TAP magn.')
ax2.errorbar(ww+0.005, means[:,8], yerr=sd[:, 8], fmt='bv', label=r'parallel MF magn.')
ax2.errorbar(ww, means[:,9], yerr=sd[:, 9], fmt='rv', label=r'parallel TAP magn.')
ax2.legend(loc=2)
ax2.set_ylim([0, 0.008])
ax2.set_ylabel('MSE')
ax2.set_xlabel(r'absolute couplings strength')
plt.show()
fig.savefig('randomWeightsRBMMAG.pdf')

n = 10  # size of layer
lattice = LatticeRBM(n)
ww = np.arange(.1, 1, step=.1)
results = np.zeros((10, 9, 12))
for itr in range(10):
    print(itr)
    res = np.zeros((0, 12))
    for weight in ww:
        W = .2 * np.random.random((n, n)) -.1  # definition
        for ii in range(n):
            for jj in range(n):
                if jj % 2:
                    W[ii, jj] += -1 * weight
                else:
                    W[ii, jj] += 1 * weight

        lattice.m = np.ones((2*n, 1)) / 2.0
        lattice.vis = np.ones((n, 1)) / 2.0; lattice.hid = np.ones((n, 1)) / 2.0
        for ii in range(20):
            updateRBM2(lattice, W, n, "naive")
        default = partitionRBM2(lattice, W, n, "naive")
        magMF = np.concatenate((lattice.hid, lattice.vis), axis=0)

        lattice.m = np.ones((2*n, 1)) / 2.0
        lattice.vis = np.ones((n, 1)) / 2.0; lattice.hid = np.ones((n, 1)) / 2.0
        for ii in range(30):
            updateRBM1(lattice, W, n, "naive")
        sequential = partitionRBM1(lattice, W, n, "naive")
        seqMF = lattice.m

        lattice.m = np.ones((2*n, 1)) / 2.0
        lattice.vis = np.ones((n, 1)) / 2.0; lattice.hid = np.ones((n, 1)) / 2.0
        for ii in range(20):
            updateRBM2(lattice, W, n, "tap")
        defaultTAP = partitionRBM2(lattice, W, n, "tap")
        magTAP = np.concatenate((lattice.hid, lattice.vis), axis=0)

        lattice.m = np.ones((2*n, 1)) / 2.0
        lattice.vis = np.ones((n, 1)) / 2.0; lattice.hid = np.ones((n, 1)) / 2.0
        for ii in range(30):
            updateRBM1(lattice, W, n, "naive")
        sequentialTAP = partitionRBM1(lattice, W, n, "tap")
        seqTAP = lattice.m

        lattice.m = np.ones((2*n, 1)) / 2.0
        lattice.vis = np.ones((n, 1)) / 2.0; lattice.hid = np.ones((n, 1)) / 2.0
        for ii in range(20):
            updateRBM3(lattice, W, n, "naive")
        parallel = partitionRBM2(lattice, W, n, "naive")
        parMF = np.concatenate((lattice.hid, lattice.vis), axis=0)

        lattice.m = np.ones((2*n, 1)) / 2.0
        lattice.vis = np.ones((n, 1)) / 2.0; lattice.hid = np.ones((n, 1)) / 2.0
        for ii in range(20):
            updateRBM3(lattice, W, n, "tap")
        parallelTAP = partitionRBM2(lattice, W, n, "tap")
        parTAP = np.concatenate((lattice.hid, lattice.vis), axis=0)

        F = -np.log(magnetizationRBM(lattice, W, n))
        magReal = copy.deepcopy(lattice.m)
        # results

        res = np.vstack((res, [abs(F - default)[0][0], abs(F - sequential)[0][0], abs(F - sequentialTAP)[0][0], abs(F - defaultTAP)[0][0], \
                               abs(F - parallel)[0][0], abs(F - parallelTAP)[0][0],\
                               magMSE(magReal, magMF)[0], magMSE(magReal, magTAP)[0],  magMSE(magReal, seqMF), magMSE(magReal, seqTAP), \
                               magMSE(magReal, parMF)[0], magMSE(magReal, parTAP)[0]]))
    results[itr] = res

np.save('saves', results)
'''

import matplotlib.pyplot as plt
results = np.load('saves.npy')
ww = np.arange(.1, 1, step=.1)
# mf - Z, mftap - Z, tap - Z,
print(results[9, 0, :])
means = np.zeros((9, 12))
sd = np.zeros((9, 12))
for ii in range(9):
    means[ii, :] = np.mean(results[:, ii, :], axis = 0)
    sd[ii, :] = np.std(results[:, ii, :], axis = 0)

fig, ax1 = plt.subplots(1)
ax1.errorbar(ww, means[:, 0], yerr=sd[:, 0], fmt='bv', label=r'defaultMF')
ax1.errorbar(ww, means[:, 3], yerr=sd[:, 3], fmt='rv', label=r'defaultTAP')
ax1.errorbar(ww + 0.01, means[:, 1], yerr=sd[:, 1], fmt='b^', label=r'sequentialMF')
ax1.errorbar(ww+ 0.01, means[:, 2], yerr=sd[:, 2], fmt='r^', label=r'sequentialTAP')
ax1.errorbar(ww + 0.005, means[:, 4], yerr=sd[:, 4], fmt='bo', label=r'parallelMF')
ax1.errorbar(ww+ 0.005, means[:, 5], yerr=sd[:, 5], fmt='ro', label=r'parallelTAP')
ax1.legend(loc=1)

ax1.set_ylabel('Abs. difference from true -log(Z)')
ax1.set_xlabel(r'absolute couplings strength')
plt.show()
fig.savefig('randomWeightsSignsRBMZ.pdf')

fig, ax2 = plt.subplots(1)
ax2.errorbar(ww, means[:,6], yerr=sd[:, 6], fmt='bo', label=r'default MF magn.')
ax2.errorbar(ww, means[:,7], yerr=sd[:, 7], fmt='ro', label=r'default TAP magn.')
ax2.errorbar(ww+ 0.01, means[:,8], yerr=sd[:, 8], fmt='b^', label=r'sequential MF magn.')
ax2.errorbar(ww+ 0.01, means[:,9], yerr=sd[:, 9], fmt='r^', label=r'sequential TAP magn.')
ax2.errorbar(ww+ 0.005, means[:,10], yerr=sd[:, 10], fmt='bv', label=r'parallel MF magn.')
ax2.errorbar(ww+ 0.005, means[:,11], yerr=sd[:, 11], fmt='rv', label=r'parallel TAP magn.')
ax2.legend(loc=1)

ax2.set_ylabel('MSE')
ax2.set_xlabel(r'absolute couplings strength')
plt.show()
fig.savefig('randomWeightsSignsRBMMAG.pdf')



