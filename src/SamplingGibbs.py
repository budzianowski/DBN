### Base definitions for Gibbs sampling
import numpy as np


def sample(means):
    samples = np.random.binomial(size=means.shape, n=1, p=means)
    return samples


def sample_hiddens(rbm, vis):
    means = rbm.probHidCondOnVis(vis)
    samples = sample(means)
    return samples, means


def sample_visibles(rbm, hid):
    means = rbm.probVisCondOnHid(hid)
    samples = sample(means)
    return samples, means


## Main course
def MCMC(rbm, init, iterations=1, StartMode="hidden"):
    if StartMode == "visible":
    # In this first mode we assume that we are starting from the visible samples. E.g. in
    # the case of binary RBM, we should be starting with binary samples.
        vis_samples = init                                    # Start from the visible samples
        vis_means = init                                    # Giving a starting point for the means
        hid_samples, hid_means = sample_hiddens(rbm, vis_samples)     # Get the first hidden means [NMF-ish]

    if StartMode == "hidden":
    # In this second mode we assume that we are starting from a set of hidden
    # samples. Because of this, we increment the iteration count by 1
        hid_samples = init
        hid_means = init

    for ii in range(iterations):
        vis_samples, vis_means = sample_visibles(rbm, hid_samples)          # Sample the visible units from true distribution
        hid_samples, hid_means = sample_hiddens(rbm, vis_samples)           # Update the hidden unit means, a NMF-ish approach

    return vis_samples, vis_means, hid_samples, hid_means

