import numpy as np
from scipy.special import expit  # fast sigmoid
import SamplingEMF
import SamplingGibbs

import copy
try:
    import cPickle as pickle
except:
    import pickle


"""Module with main class of a package RBM which enables to define and
compute basic statistics for the restricted Boltzmann structure.
"""


class RBM(object):
    """Restricted Boltzmann Machine (RBM)"""
    def __init__(self, params=None, n_vis=784, n_hid=500, sigma=0.01, momentum=0.0, trainData=None, wiseStart=False):
        """
        RBM constructor. Defines the parameters of the model along with
        basic operations for inferring hidden from visible (and vice-versa),
        as well as for computing model statistics.

        :param params: if model is loaded stores all weights values.

        :param n_vis: number of visible units

        :param n_hid: number of hidden units

        :arg W: matrix of coupling parameters

        :arg hbias: hidden bias vector

        :arg vbias: visible bias vector

        :arg dW: current gradient on the coupling parameters

        :arg persistent_chain_hid: hidden persistent chain

        :arg persistent_chain_vis: visible persistent chain
        """

        self.eps = 1e-6  # Some "tiny" value, used to enforce min/max boundary conditions
        self.momentum = momentum
        self.layer = 1 # for DBN
        if params is None:
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

            # Initialization of visual bias - Hinton's recommendation
            self.vbias = np.zeros((n_vis, 1))
            if trainData is not None:
                temp = np.mean(trainData, 1)
                np.clip(temp, self.eps, 1 - self.eps, out=temp)
                self.vbias = np.log(temp / (1 - temp)).reshape((n_vis, 1))
        else:
            self.W                      = params[0]
            self.W2                     = params[1]
            self.W3                     = params[2]
            self.vbias                  = params[3]
            self.hbias                  = params[4]
            self.dW                     = params[5]
            self.dW_prev                = params[6]
            self.persistent_chain_vis   = params[7]
            self.persistent_chain_hid   = params[8]

    def passHidToVis(self, hid):
        """ This function propagates the hidden units activation downwards to
        the visible units """
        return np.dot(self.W.T, hid) + self.vbias

    def passVisToHid(self, vis):
        """ Propagates the visible units activation upwards to
        the hidden units """
        return np.dot(self.W, vis) + self.hbias

    def probHidCondOnVis(self, vis):
        """ Function to compute probability p(h|v) """
        return expit(self.passVisToHid(vis))

    def probVisCondOnHid(self, hid):
        """ Function to compute probability p(v|h) """
        return expit(self.passHidToVis(hid))

    def free_energy(self, vis):
        """ Computes the clamped free energy """
        vb = np.dot(self.vbias.T, vis)
        Wv_b = np.sum(np.log(1 + np.exp(np.dot(self.W, vis) + self.hbias)), axis=0)
        return - vb - Wv_b

    def score_samples(self, vis):
        """ Computes proxy LL """
        n_feat, n_samples = vis.shape
        vis_corrupted = np.copy(vis)
        idxs = np.random.random_integers(0, n_feat - 1, n_samples)
        for (i, j) in zip(idxs, range(n_samples)):  # corruption of particular bit in a given (j) sample
            vis_corrupted[i, j] = 1 - vis_corrupted[i, j]

        fe = self.free_energy(vis)
        fe_corrupted = self.free_energy(vis_corrupted)
        logPL = n_feat * np.log(expit(fe_corrupted - fe))

        return logPL

    def recon_error(self, vis):
        """ Computes reconstruction error """
        # Fully forward MF operation to get back to visible samples
        vis_rec = self.probVisCondOnHid(self.probHidCondOnVis(vis))
        # Get the total error over the whole tested visible set,
        # here, as MSE
        mse = np.sum(vis * np.log(vis_rec) + (1 - vis) * np.log(1 - vis_rec), 0)
        return mse

    def score_samples_TAP(self, vis, n_iter=10, approx="tap2"):
        """ Computes Gibbs free energy """
        m_vis, m_hid = SamplingEMF.iter_mag(self, vis, iterations=n_iter, approx=approx)
        # clipping to compute entropy
        m_vis = np.clip(m_vis, self.eps, 1 - self.eps)
        m_hid = np.clip(m_hid, self.eps, 1 - self.eps)

        m_vis2 = m_vis ** 2
        m_hid2 = m_hid ** 2

        Entropy = np.sum(m_vis * np.log(m_vis) + (1.0 - m_vis) * np.log(1.0 - m_vis), 0) \
                  + np.sum(m_hid * np.log(m_hid) + (1.0 - m_hid) * np.log(1.0 - m_hid), 0)
        Naive = np.sum(self.vbias * m_vis, 0) + np.sum(self.hbias * m_hid, 0) + \
                np.sum(m_hid * np.dot(self.W, m_vis), 0)
        Onsager = 0.5 * np.sum((m_hid-m_hid2) * np.dot(self.W**2, m_vis-m_vis2), 0)

        fe_tap = Entropy - Naive - Onsager
        if approx == "tap3":
            visible = (m_vis-m_vis2) * (0.5 - m_vis)
            hidden = (m_hid-m_hid2) * (0.5 - m_hid)
            fe_tap -= (2.0 / 3.0) * np.sum(hidden * np.dot(self.W**3, visible), 0)

        fe = self.free_energy(vis)
        return -fe + fe_tap

    def baseModel(self, vis):  # ML model
        counts = np.sum(vis, axis=1).reshape(784, 1)
        p = (counts + 5*1) / (vis.shape[1] + 5*1)  # divided by number of possible vales - 100 in our case plus our prior
        vbias = np.log(p) - np.log(1 - p)  # logit function

        return vbias

    def reconstructionArray(self, vis):
        import matplotlib.pyplot as plt
        import PIL.Image
        print('plotting original')
        n = vis.shape[1]
        for ii in range(n):
            plt.matshow(vis[:, ii].reshape(28, 28), fignum=100, cmap=plt.cm.gray)
            name = 'org' + str(ii) + '.jpg'
            plt.savefig(name)

        list_im = []
        for jj in range(n):
            list_im.append('org' + str(jj) + '.jpg')
        imgs    = [PIL.Image.open(i) for i in list_im]
        imgs_comb = np.hstack((np.asarray(i) for i in imgs))
        horizontal = []
        horizontal.append(imgs_comb)

        imgs_comb = np.vstack((np.asarray(i) for i in horizontal))
        imgs_comb = PIL.Image.fromarray(imgs_comb)
        imgs_comb.save('recon' + '.jpg')

    def reconstruction(self, vis):
        import matplotlib.pyplot as plt
        print('plotting original')
        plt.matshow(vis.reshape(28, 28), fignum=100, cmap=plt.cm.gray)
        plt.savefig('org.pdf')
        temp = np.copy(vis).reshape(784, 1)
        prob = np.dot(self.W, temp) + self.hbias
        prob = expit(prob)
        prob = np.dot(self.W.T, prob) + self.vbias
        prob = expit(prob)
        print('plotting reconstruction')
        plt.matshow(prob.reshape(28, 28), fignum=100, cmap=plt.cm.gray)
        plt.savefig('rec.pdf')

    def AIS(self, vbias=None):
        # the helping distribution has the form exp(-beta * v) zero weight matrix
        n_hid = self.hbias.shape[0]
        n_vis = self.vbias.shape[0]

        # ML estimate for visbias
        if vbias:
            visbias_base = vbias
        else:
            visbias_base = self.vbias

        negdata = 1. / (1. + np.exp(-visbias_base))  # p_a(v)
        negdata = np.random.binomial(size=(n_vis, 1), n=1, p=negdata)
        logww = - (np.dot(visbias_base.T, negdata))   # start of AIS estimate only auxiliary distribution
        Bv_base = np.dot(visbias_base.T, negdata)

        # start
        Wh = np.dot(self.W, negdata) + self.hbias
        Bv = np.dot(self.vbias.T, negdata)
        betas = np.linspace(0.001, .5, 1000)
        betas = np.append(betas, np.linspace(0.5, .899, 4000))
        betas = np.append(betas, np.linspace(0.9, .999, 5000))

        for beta in betas:  # without beta = 0 and 1
            expWh = np.exp(beta * Wh)
            logww += (1 - beta) * Bv_base + beta * Bv + np.sum(np.log(1 + expWh))  # p_k^* (v)

            # transitions
            poshidprobs = expWh / (1. + expWh)  # sigmoid differently
            poshidstates = np.random.binomial(size=(n_hid, 1), n=1, p=poshidprobs)

            negdata = 1./(1 + np.exp(-(1 - beta) * visbias_base - beta * (np.dot(self.W.T, poshidstates) + self.vbias)));
            negdata = np.random.binomial(size=(n_vis, 1), n=1, p=negdata)

            # update
            Bv_base = np.dot(visbias_base.T, negdata)
            Wh = np.dot(self.W, negdata) + self.hbias
            Bv = np.dot(self.vbias.T, negdata)

            expWh = np.exp(beta * Wh)
            logww -= (1 - beta) * Bv_base + beta * Bv + np.sum(np.log(1 + expWh))  # p_k^* (v)

        # last update
        beta = 1
        expWh = np.exp(beta * Wh)
        logww += beta * Bv + np.sum(np.log(1 + expWh))  # p_k^* (v)

        Z_A = np.sum(np.log(1 + np.exp(visbias_base)))   # Z_A = np.prod(1 + np.exp(self.vbias))
        Z_B = logww + Z_A

        return Z_A, Z_B

    def save(self, file_name):
        """ Function to save all main parameters """
        print('Saving model to: {0}'.format(file_name))

        with open(file_name, 'wb') as f:
            pickle.dump(self.W, f)
            pickle.dump(self.W2, f)
            pickle.dump(self.W3, f)
            pickle.dump(self.vbias, f)
            pickle.dump(self.hbias, f)
            pickle.dump(self.dW, f)
            pickle.dump(self.dW_prev, f)
            pickle.dump(self.persistent_chain_vis, f)
            pickle.dump(self.persistent_chain_hid, f)

    @staticmethod
    def load(file_name):
        """ Function to load paramaters from numpy pickle """
        print('Loading model form : {0}'.format(file_name))
        params = []
        with open(file_name, 'rb') as f:
            while True:
                try:
                    p = pickle.load(f)
                    params.append(p)
                except EOFError:
                    break
        return params
