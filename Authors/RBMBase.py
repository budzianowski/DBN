import numpy as np
from scipy.special import expit  # fast sigmoid

"""
    # Boltzmann.RBM{V,H} (RBMBase.jl)
    ## Description
        A structure for containing all of the restricted Boltzmann Machine (RBM)
        model parameters. Besides just the model parameters (couplings, biases),
        the structure also contains variables which are pertinent to the RBM training
        procedure.
    ## Structure
        - `W::Matrix{Float64}`:       The matrix of coupling parameters (RBM model parameter)
        - `W2::Matrix{Float64}`:      The square of `W` (used for EMF learning)
        - `W3::Matrix{Float64}`:      The cube of `W` (used for EMF learning)
        - `vbias::Vector{Float64}`:   The visible unit biases (RBM model parameter)
        - `hbias::Vector{Float64}`:   The hidden unit biases (RBM model parameter)
        - `dW::Matrix{Float64}`:      The current gradient on the coupling parameters (used for RBM training)
        - `dW_prev::Matrix{Float64}`: The last gradient on the coupling parmaters (used for RBM training)
        - `persistent_chain_vis::Matrix{Float64}`: Visible fantasy particles (used for RBM persistent mode training)
        - `persistent_chain_hid::Matrix{Float64}`: Hidden fantasy particles (used for RBM persistent mode training)
        - `momentum::Float64`:        Amount of last gradient to add to the current gradient (used for RBM training)
        - `VisShape::Tuple{Int,Int}`: Final output shape of the visible units
"""

class RBM(object):
    def __init__(self, n_vis, n_hid, sigma=0.01, momentum=0.0, TrainData=None, wiseStart=False, batchSize=100):

        # Initialize the weighting matrix by drawing from an iid Gaussian
        # of the specified standard deviation.
        if wiseStart:
            self.W = np.random.uniform(
                    low=-4 * np.sqrt(6. / (n_hid + n_vis)),
                    high=4 * np.sqrt(6. / (n_hid + n_vis)),
                    size=(n_hid, n_vis)
                )
        else:
            self.W = np.random.normal(0, sigma, (n_hid, n_vis))

        self.W2 = np.random.normal(0, sigma, (n_hid, n_vis))
        self.W3 = np.random.normal(0, sigma, (n_hid, n_vis))

        self.hbias = np.zeros((n_hid, 1))

        self.dW = np.zeros((n_hid, n_vis))
        self.dW_prev = np.zeros((n_hid, n_vis))

        self.persistent_chain_vis = None #np.zeros((n_vis, batchSize)) # TODO should be size of the batch
        self.persistent_chain_hid = None

        self.eps = 1e-8  # Some "tiny" value, used to enforce min/max boundary conditions
        self.momentum = momentum
        self.idxs = 0  # index for computing proxy LL

        # If the user specifies the training dataset, it can be useful to
        # initialize the visible biases according to the empirical expected
        # feature values of the training data.
        # TODO: Generalize this biasing. Currently, the biasing is only written for
        #       the case of binary RBMs.

        # Initialization of visual bias - Hinton's recommendation
        self.vbias = np.zeros((n_vis, 1))
        if TrainData is not None:
            temp = np.mean(TrainData, 1)
            np.clip(temp, self.eps, 1 - self.eps, out=temp)
            self.vbias = np.log(temp / (1 - temp)).reshape((n_vis, 1))


def PassHidToVis(rbm, hid):
    return np.dot(rbm.W.T, hid) + rbm.vbias


def PassVisToHid(rbm, vis):
    return np.dot(rbm.W, vis) + rbm.hbias


def ProbHidCondOnVis(rbm, vis):
    return expit(PassVisToHid(rbm, vis))


def ProbVisCondOnHid(rbm, hid):
    return expit(PassHidToVis(rbm, hid))


# # TODO: check it
# def PinField!(rbm:,pinning_field::Vec{Float64})
#     pos_inf_locations = pinning_field > 0
#     neg_inf_locations = pinning_field < 0
#
#     rbm.vbias(pos_inf_locations) =  Inf
#     rbm.vbias(neg_inf_locations) = -Inf
