import numpy as np
from scipy.special import expit
import RBMBase as RBMBase

def sample(means):
    samples = np.random.binomial(size=means.shape, n=1, p=means)
    return samples


def sample_hiddens(rbm, vis):
    means = RBMBase.ProbHidCondOnVis(rbm, vis)
    samples = sample(means)
    return samples, means


def sample_visibles(rbm, hid):
    means = RBMBase.ProbVisCondOnHid(rbm, hid)
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
        #iterations += 1

    for ii in range(iterations):
        vis_samples, vis_means = sample_visibles(rbm, hid_samples)          # Sample the visible units from true distribution
        hid_samples, hid_means = sample_hiddens(rbm, vis_samples)           # Update the hidden unit means, a NMF-ish approach

    return vis_samples, vis_means, hid_samples, hid_means



## MAIN COURSE
# def gibbs(rbm, vis, n_times=1):
#     v_pos = vis
#     h_samp, h_pos = sample_hiddens(rbm, v_pos)
#     h_neg = Array(Float64,0,0)::Mat{Float64}
#     v_neg = Array(Float64,0,0)::Mat{Float64}
#     if n_times > 0
#     # Save computation by setting `n_times=0` in the case
#     # of persistent CD.
#         v_neg = sample_visibles(rbm, h_samp)
#         h_samp, h_neg = sample_hiddens(rbm, v_neg)
#         for i=1:n_times-1
#             v_neg = sample_visibles(rbm, h_samp)
#             h_samp, h_neg = sample_hiddens(rbm, v_neg)
#         end
#     end
#     return v_pos, h_pos, v_neg, h_neg

