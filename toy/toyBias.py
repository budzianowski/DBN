#!/usr/bin/env python
#

import numpy as np
from math import floor
from scipy.special import expit

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def whatState(state, n):  # states [ 0, ..., n-1]
    if state > (n ** 2 - 1):
        print('to big')
    else:
        ii = floor(state / n)
        jj = state - ii * n
    return ii, jj


class Node(object):
    Z = 0  # partition value
    def __init__(self):
        self.state = np.random.binomial(n=1, p=0.5)
        self.m = np.random.random()  # magnetization
        self.b = np.random.random() - .5  # random bias

def energy(lattice, W, n):
    E = 0
    for ss in range(0, n ** 2):  # over states
        ii, jj = whatState(ss, n)
        E -= 0.5 * W[ii][jj][0] * lattice[ii][jj].state * lattice[ii][jj-1].state  # W w * s_i *s_j
        E -= 0.5 * W[ii][jj][1] * lattice[ii][jj].state * lattice[ii][(jj+1) % n].state  # E
        E -= 0.5 * W[ii][jj][2] * lattice[ii][jj].state * lattice[ii-1][jj].state  # N
        E -= 0.5 * W[ii][jj][3] * lattice[ii][jj].state * lattice[(ii+1) % n][jj].state  # S
        E -= lattice[ii][jj].b * lattice[ii][jj].state
    return np.exp(-E)


def newSet(x):
    for i in range(len(x)):
        x[i] += 1
        if x[i] <= 1:
            return True
        x[i] = 0
    return False


def magnetization(lattice, W, n):
    mag = [0 for ii in range(n**2)]
    Z = 0
    x = np.zeros(n**2, dtype=int)  # number of states

    while True:
        for state in range(n**2):  # update of states
            ii, jj = whatState(state, n)
            lattice[ii][jj].state = x[state]  # compute here energy

        E = energy(lattice, W, n)
        Z += E
        for state in range(n**2):
            mag[state] += E * x[state]  # x[m] is a realization of a given state

        if not newSet(x):
            break

    for state in range(n**2):
        ii, jj = whatState(state, n)
        lattice[ii][jj].m = mag[state] / Z

    return Z


def printStates(lattice, W, n):
    for state in range(n**2):
        ii, jj = whatState(state, n)
        print(lattice[ii][jj].m)


def reset(lattice, W, n):
    for state in range(n**2):
        ii, jj = whatState(state, n)
        lattice[ii][jj].m = np.random.random()


# MF or TAP
def updateMF(lattice, W, n):
    damp = 0.5
    for state in range(n**2):  # asynchronous update
        temp = 0
        ii, jj = whatState(state, n)
        temp += W[ii][jj][0] * lattice[ii][jj-1].m
        temp += W[ii][jj][1] * lattice[ii][(jj+1) % n].m
        temp += W[ii][jj][2] * lattice[ii-1][jj].m
        temp += W[ii][jj][3] * lattice[(ii+1) % n][jj].m
        lattice[ii][jj].m = damp * sigmoid(temp + lattice[ii][jj].b) + (1 - damp) * lattice[ii][jj].m


def updateTAP2(lattice, W, n):
    damp = 0.5
    for state in range(n**2):
        temp = 0
        ii, jj = whatState(state, n)
        temp += W[ii][jj][0] * lattice[ii][jj-1].m \
                - (W[ii][jj][0] ** 2) * (lattice[ii][jj].m - 0.5) * (lattice[ii][jj-1].m - lattice[ii][jj-1].m**2)# W w * s_i *s_j
        temp += W[ii][jj][1] * lattice[ii][(jj+1) % n].m  \
                - (W[ii][jj][1] ** 2) * (lattice[ii][jj].m - 0.5) * (lattice[ii][(jj+1) % n].m - lattice[ii][(jj+1) % n].m**2)# E
        temp += W[ii][jj][2] * lattice[ii-1][jj].m \
                - (W[ii][jj][2] ** 2) * (lattice[ii][jj].m - 0.5) * (lattice[ii-1][jj].m - lattice[ii-1][jj].m**2)# N
        temp += W[ii][jj][3] * lattice[(ii+1) % n][jj].m  \
                - (W[ii][jj][3] ** 2) * (lattice[ii][jj].m - 0.5) * (lattice[(ii+1) % n][jj].m - lattice[(ii+1) % n][jj].m**2)# S
        lattice[ii][jj].m = damp * sigmoid(temp + lattice[ii][jj].b) + (1 - damp) * lattice[ii][jj].m


def partitionMF(lattice, W, n):
    entropy = 0
    naive = 0
    bias = 0
    for state in range(n**2):
        ii, jj = whatState(state, n)
        bias -= lattice[ii][jj].b * lattice[ii][jj].m
        entropy += lattice[ii][jj].m * np.log(lattice[ii][jj].m) + (1 - lattice[ii][jj].m) * np.log(1 - lattice[ii][jj].m)
        naive -= W[ii][jj][0] * lattice[ii][jj].m * lattice[ii][jj-1].m
        naive -= W[ii][jj][1] * lattice[ii][jj].m * lattice[ii][(jj+1) % n].m
        naive -= W[ii][jj][2] * lattice[ii][jj].m * lattice[ii-1][jj].m
        naive -= W[ii][jj][3] * lattice[ii][jj].m * lattice[(ii+1) % n][jj].m
    return bias + entropy + (naive / 2.0)


def partitionTAP(lattice, W, n):
    entropy = 0
    naive = 0
    bias = 0
    tap = 0
    for state in range(n**2):
        ii, jj = whatState(state, n)
        entropy += lattice[ii][jj].m * np.log(lattice[ii][jj].m) + (1 - lattice[ii][jj].m) * np.log(1 - lattice[ii][jj].m)
        bias -= lattice[ii][jj].b * lattice[ii][jj].m
        naive -= W[ii][jj][0] * lattice[ii][jj-1].m
        tap -= (W[ii][jj][0] ** 2 / 2.0) * (lattice[ii][jj].m - lattice[ii][jj].m ** 2) * (lattice[ii][jj-1].m - lattice[ii][jj-1].m**2)# W w * s_i *s_j
        naive -= W[ii][jj][1] * lattice[ii][(jj+1) % n].m
        tap -= (W[ii][jj][1] ** 2 / 2.0) * (lattice[ii][jj].m - lattice[ii][jj].m ** 2) * (lattice[ii][(jj+1) % n].m - lattice[ii][(jj+1) % n].m**2)# E
        naive -= W[ii][jj][2] * lattice[ii-1][jj].m
        tap -= (W[ii][jj][2] ** 2 / 2.0) * (lattice[ii][jj].m - lattice[ii][jj].m ** 2) * (lattice[ii-1][jj].m - lattice[ii-1][jj].m**2)# N
        naive -= W[ii][jj][3] * lattice[(ii+1) % n][jj].m
        tap -= (W[ii][jj][3] ** 2 / 2.0) * (lattice[ii][jj].m - lattice[ii][jj].m ** 2) * (lattice[(ii+1) % n][jj].m - lattice[(ii+1) % n][jj].m**2)# S
    return entropy + bias + naive / 2.0 + tap / 2.0


def partition(lattice, W, n, method="tap2"):
    if method == "tap2":
        return partitionTAP(lattice, W, n)
    elif method == "naive":
        return partitionMF(lattice, W, n)


def magMSE(real, m2):
    MSE = 0
    for ii, jj in zip(real, m2):
        MSE += (ii - jj) ** 2
    MSE /= len(real)
    return MSE


def getMag(lattice, n):
    mag = []
    for state in range(n**2):
        ii, jj = whatState(state, n)
        mag.append(lattice[ii][jj].m)
    return mag


class LatticeRBM(object):
    Z = 0  # partition

    def __init__(self, n):
        self.states = np.random.binomial(n=1, p=0.5, size=(2*n, 1))
        self.m = np.ones((2*n, 1)) / 2.0  # magnetization
        self.vis = np.ones((n, 1)) / 2.0  # magnetization
        self.hid = np.ones((n, 1)) / 2.0  # magnetization
        self.b = np.ones(shape=(2 * n, 1)) * .2


def energyRBM(states, W, n, lattice):
    E = states[0:n].T.dot(np.dot(W, states[n:]))
    E += np.dot(lattice.b.T, states)
    return np.exp(E)


def magnetizationRBM(lattice, W, n):
    mag = np.zeros((2*n))  # [0 for ii in 2*n]
    Z = 0
    x = np.zeros(2*n, dtype=int)  # number of states
    while True:
        E = energyRBM(x, W, n, lattice)
        Z += E
        mag += E * x

        if not newSet(x):
            break

    lattice.m = mag / Z

    return Z


def updateRBM1(lattice, W, n, method="naive"):
    damp = 0.5
    if method == "naive":
        for ii in range(n):
            lattice.m[ii] = damp * expit(np.dot(W[ii], lattice.m[n:]) + lattice.b[ii]) + (1.0 - damp) * lattice.m[ii]
            lattice.m[n + ii] = damp * expit(np.dot(W.T[ii], lattice.m[0:n] + lattice.b[n + ii])) + (1.0 - damp) * lattice.m[n + ii]
    else:
        for ii in range(n):  # TODO sequential
            lattice.m[ii] = damp * expit(np.dot(W[ii], lattice.m[n:] + lattice.b[ii]) - (lattice.m[ii] - 0.5) * np.dot(W[ii] ** 2, lattice.m[n:] - lattice.m[n:] ** 2)) + (1.0 - damp) * lattice.m[ii]
            lattice.m[n + ii] = damp * expit(np.dot(W.T[ii], lattice.m[0:n] + lattice.b[n + ii]) - (lattice.m[n + ii] - 0.5) * np.dot(W.T[ii] ** 2, lattice.m[0:n] - lattice.m[0:n] ** 2)) + (1.0 - damp) * lattice.m[n + ii]


def updateRBM2(lattice, W, n, method="naive"):
    damp = 0.5
    if method == "naive":
        lattice.hid = damp * expit(np.dot(W, lattice.vis) + lattice.b[:n]) + (1.0 - damp) * lattice.hid
        lattice.vis = damp * expit(np.dot(W.T, lattice.hid) + lattice.b[n:]) + (1.0 - damp) * lattice.vis
    else:
        lattice.hid = damp * expit(np.dot(W, lattice.vis) + lattice.b[:n] - (lattice.hid - .5) * np.dot((W ** 2), (lattice.vis - lattice.vis ** 2))) + (1.0 - damp) * lattice.hid
        lattice.vis = damp * expit(np.dot(W.T, lattice.hid) + lattice.b[n:] - (lattice.vis - .5) * np.dot((W.T ** 2), (lattice.hid - lattice.hid ** 2))) + (1.0 - damp) * lattice.vis


def updateRBM3(lattice, W, n, method="naive"):  # parallel
    damp = 0.5
    if method == "naive":
        hid = damp * expit(np.dot(W, lattice.vis) + lattice.b[:n]) + (1.0 - damp) * lattice.hid
        vis = damp * expit(np.dot(W.T, lattice.hid) + lattice.b[n:]) + (1.0 - damp) * lattice.vis
    else:
        hid = damp * expit(np.dot(W, lattice.vis) + lattice.b[:n] - (lattice.hid - .5) * np.dot((W ** 2), (lattice.vis - lattice.vis ** 2))) + (1.0 - damp) * lattice.hid
        vis = damp * expit(np.dot(W.T, lattice.hid) + lattice.b[n:] - (lattice.vis - .5) * np.dot((W.T ** 2), (lattice.hid - lattice.hid ** 2))) + (1.0 - damp) * lattice.vis

    lattice.hid = hid
    lattice.vis = vis


def partitionRBM1(lattice, W, n, method="naive"):
    temp = np.sum(lattice.m * np.log(lattice.m) + (1.0 - lattice.m) * np.log(1.0 - lattice.m))
    temp -= np.dot(lattice.b.T, lattice.m)  # bias
    temp -= lattice.m[0:n].T.dot(np.dot(W, lattice.m[n:]))
    if method == "tap":
        temp -= (lattice.m[0:n] - lattice.m[0:n] ** 2).T.dot(np.dot((W ** 2) / 2.0, (lattice.m[n:] - lattice.m[n:] ** 2)))

    return temp


def partitionRBM2(lattice, W, n, method="naive"):
    temp = np.sum(lattice.vis * np.log(lattice.vis) + (1.0 - lattice.vis) * np.log(1.0 - lattice.vis))
    temp += np.sum(lattice.hid * np.log(lattice.hid) + (1.0 - lattice.hid) * np.log(1.0 - lattice.hid))
    mag = np.concatenate((lattice.hid, lattice.vis), axis=0)
    temp -= np.dot(lattice.b.T, mag)  # bias
    temp -= lattice.hid.T.dot(np.dot(W, lattice.vis))
    if method == "tap":
        temp -= (lattice.hid - lattice.hid ** 2).T.dot(np.dot((W ** 2) / 2.0, (lattice.vis - lattice.vis ** 2)))

    return temp


