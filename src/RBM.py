import numpy as np
from scipy.special import expit  # fast sigmoid

from scipy.special import expit
import SamplingEMF
import SamplingGibbs
import copy


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

        self.persistent_chain_vis = None
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

    def passHidToVis(self, hid):
        return np.dot(self.W.T, hid) + self.vbias

    def passVisToHid(self, vis):
        return np.dot(self.W, vis) + self.hbias

    def probHidCondOnVis(self, vis):
        return expit(self.passVisToHid(vis))

    def probVisCondOnHid(self, hid):
        return expit(self.passHidToVis(hid))

    def free_energy(self, vis):
        vb = np.dot(self.vbias.T, vis)
        Wx_b_log = np.sum(np.log(1 + np.exp(self.hbias + np.dot(self.W, vis))), axis=0)
        return - vb - Wx_b_log

    def score_samples(self, vis):
        n_feat, n_samples = vis.shape
        vis_corrupted = copy.deepcopy(vis)
        idxs = np.random.random_integers(0, n_feat - 1, n_samples)
        for (i, j) in zip(idxs, range(n_samples)):  # corruption of particular bit in a given (j) sample
            vis_corrupted[i, j] = 1 - vis_corrupted[i, j]

        fe = self.free_energy(vis)
        fe_corrupted = self.free_energy(vis_corrupted)

        logPL = n_feat * np.log(expit(fe_corrupted - fe))

        return logPL

    def recon_error(self, vis):
        # Fully forward MF operation to get back to visible samples
        vis_rec = self.probVisCondOnHid(self.probHidCondOnVis(vis))
        # Get the total error over the whole tested visible set,
        # here, as MSE # change the difference!
        mse = np.sum(vis * np.log(vis_rec) + (1 - vis) * np.log(1 - vis_rec), 0)

        return mse

    def score_samples_TAP(self, vis, n_iter=5, approx="tap2"):
        m_vis, m_hid = SamplingEMF.iter_mag(self, vis, iterations=n_iter, approx=approx)
        eps = 1e-6
        # clipping to compute entropy
        m_vis = np.clip(m_vis, eps, 1 - eps)
        m_hid = np.clip(m_hid, eps, 1 - eps)

        m_vis2 = m_vis ** 2
        m_hid2 = m_hid ** 2

        Entropy = np.sum(m_vis * np.log(m_vis) + (1.0 - m_vis) * np.log(1.0 - m_vis), 0) \
                  + np.sum(m_hid * np.log(m_hid) + (1.0 - m_hid) * np.log(1.0 - m_hid), 0)
        Naive = np.sum(self.vbias * m_vis, 0) + np.sum(self.hbias * m_hid, 0) + \
                np.sum(m_hid * np.dot(self.W, m_vis), 0)
        Onsager = 0.5 * np.sum((m_hid-m_hid2) * np.dot(self.W2, m_vis-m_vis2), 0)

        fe_tap = Entropy - Naive - Onsager

        if "tap3" in approx:
            # TODO third term
            pass
            fe_tap += 0

        fe = self.free_energy(vis)

        return -fe + fe_tap

# # TODO: check it
# def PinField!(rbm:,pinning_field::Vec{Float64})
#     pos_inf_locations = pinning_field > 0
#     neg_inf_locations = pinning_field < 0
#
#     rbm.vbias(pos_inf_locations) =  Inf
#     rbm.vbias(neg_inf_locations) = -Inf
