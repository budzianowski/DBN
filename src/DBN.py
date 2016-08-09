import numpy as np
from scipy.special import expit  # fast sigmoid
import SamplingEMF
import SamplingGibbs

try:
    import cPickle as pickle
except:
    import pickle


class DBN(object):
    """Deep Belief Nets (RBM)"""
    def __init__(self, layer=1, params=None, n_vis=784, n_hid=500, sigma=0.01, momentum=0.0, trainData=None, wiseStart=False):
        self.eps = 1e-6  # Some "tiny" value, used to enforce min/max boundary conditions
        self.momentum = momentum
        self.layer = layer

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
            if self.layer == 2:
                self.W1                      = params[0]
                self.vbias1                  = params[3]
                self.hbias1                  = params[4]
            if self.layer == 3:
                self.W1                      = params[3]
                self.vbias1                  = params[4]
                self.hbias1                  = params[5]
                self.W22                     = params[0]  # coupling in the second layer
                self.vbias2                  = params[1]
                self.hbias2                  = params[2]
            if self.layer == 4:
                self.W1                      = params[3]
                self.vbias1                  = params[4]
                self.hbias1                  = params[5]
                self.W22                     = params[6]
                self.vbias2                  = params[7]
                self.hbias2                  = params[8]
                self.W33                     = params[0]  # coupling in the third layer
                self.vbias3                  = params[1]
                self.hbias3                  = params[2]

            # default layer
            self.W = np.random.normal(0, sigma, (n_hid, n_vis))
            self.W2 = np.random.normal(0, sigma, (n_hid, n_vis))
            self.W3 = np.random.normal(0, sigma, (n_hid, n_vis))
            self.hbias = np.zeros((n_hid, 1))
            self.vbias = np.zeros((n_vis, 1))
            self.dW = np.zeros((n_hid, n_vis))
            self.dW_prev = np.zeros((n_hid, n_vis))
            self.persistent_chain_vis = None
            self.persistent_chain_hid = None

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

    # Passing through first layer
    def probHidCondOnVis1Layer(self, vis):
        """ Function to compute probability p(h|v) - 1 layer """
        Wv_h = np.dot(self.W1, vis) + self.hbias1
        return expit(Wv_h)

    # Passing through second layer
    def probHidCondOnVis2Layer(self, vis):
        """ Function to compute probability p(h|v) - 2 layer """
        Wv_h = np.dot(self.W22, vis) + self.hbias2
        return expit(Wv_h)

    def free_energy(self, vis):
        """ Computes the clamped free energy """
        vb = np.dot(self.vbias.T, vis)
        Wv_b = np.sum(np.log(1 + np.exp(np.dot(self.W, vis) + self.hbias)), axis=0)
        return - vb - Wv_b

    def score_samples(self, vis):
        """ Computes proxy LL """
        prob = vis
        if self.layer >= 2:
            prob = self.probHidCondOnVis1Layer(vis)
            if self.layer == 3:
                prob = self.probHidCondOnVis2Layer(prob)
        n_feat, n_samples = prob.shape
        vis_corrupted = np.copy(prob)
        idxs = np.random.random_integers(0, n_feat - 1, n_samples)
        for (i, j) in zip(idxs, range(n_samples)):  # corruption of particular bit in a given (j) sample
            vis_corrupted[i, j] = 1 - vis_corrupted[i, j]

        fe = self.free_energy(prob)
        fe_corrupted = self.free_energy(vis_corrupted)
        logPL = n_feat * np.log(expit(fe_corrupted - fe))

        return logPL

    def score_samples_TAP(self, vis, n_iter=5, approx="tap2"):
        prob = vis
        if self.layer >= 2:
            prob = self.probHidCondOnVis1Layer(vis)
            if self.layer == 3:
                prob = self.probHidCondOnVis2Layer(prob)
        """ Computes Gibbs free energy """
        m_vis, m_hid = SamplingEMF.iter_mag(self, prob, iterations=n_iter, approx="tap2")
        # clipping to compute entropy
        m_vis = np.clip(m_vis, self.eps, 1 - self.eps)
        m_hid = np.clip(m_hid, self.eps, 1 - self.eps)

        m_vis2 = m_vis ** 2
        m_hid2 = m_hid ** 2

        Entropy = np.sum(m_vis * np.log(m_vis) + (1.0 - m_vis) * np.log(1.0 - m_vis), 0) \
                  + np.sum(m_hid * np.log(m_hid) + (1.0 - m_hid) * np.log(1.0 - m_hid), 0)
        Naive = np.sum(self.vbias * m_vis, 0) + np.sum(self.hbias * m_hid, 0) + \
                np.sum(m_hid * np.dot(self.W, m_vis), 0)
        Onsager = 0.5 * np.sum((m_hid-m_hid2) * np.dot(self.W2, m_vis-m_vis2), 0)

        fe_tap = Entropy - Naive - Onsager

        if "tap3" in approx:
            visible = (m_vis-m_vis2) * (0.5 - m_vis)
            hidden = (m_hid-m_hid2) * (0.5 - m_hid)
            fe_tap -= (2.0 / 3.0) * np.sum(hidden * np.dot(self.W3, visible), 0)

        fe = self.free_energy(prob)
        return -fe + fe_tap

    def recon_error(self, vis):
        """ Computes reconstruction error """
        prob = vis
        if self.layer >= 2:
            prob = self.probHidCondOnVis1Layer(vis)
            if self.layer == 3:
                prob = self.probHidCondOnVis2Layer(prob)
        # Fully forward MF operation to get back to visible samples
        vis_rec = self.probVisCondOnHid(self.probHidCondOnVis(prob))
        # Get the total error over the whole tested visible set,
        # here, as MSE
        mse = np.sum(prob * np.log(vis_rec) + (1 - prob) * np.log(1 - vis_rec), 0)

        return mse

    def baseModel(self, vis):  # ML model
        counts = np.sum(vis, axis=1).reshape(784, 1)
        p = (counts + 5*1) / (vis.shape[1] + 5*1)  # divided by number of possible vales - 100 in our case plus our prior
        vbias = np.log(p) - np.log(1 - p)  # logit function

        return vbias

    def reconstructionArray(self, vis):
        import matplotlib.pyplot as plt
        import PIL.Image
        horizontal = []
        n = vis.shape[1]
        for ii in range(n):
            plt.matshow(vis[:, ii].reshape(28, 28), fignum=100, cmap=plt.cm.gray)
            name = 'org' + str(ii) + '.jpg'
            plt.savefig(name)
            temp = np.copy(vis[:, ii]).reshape(784, 1)
            prob = np.dot(self.W1, temp) + self.hbias1
            prob = expit(prob)
            if self.layer >= 3:
                prob = np.dot(self.W22, prob) + self.hbias2
                prob = expit(prob)
                if self.layer == 4:
                    prob = np.dot(self.W33, prob) + self.hbias3
                    prob = expit(prob)
                # decoder
                    prob = np.dot(self.W33.T, prob) + self.vbias3
                    prob = expit(prob)
                prob = np.dot(self.W22.T, prob) + self.vbias2
                prob = expit(prob)
            prob = np.dot(self.W1.T, prob) + self.vbias1
            prob = expit(prob)
            plt.matshow(prob.reshape(28, 28), fignum=100, cmap=plt.cm.gray)
            name = 'rec' + str(ii) + '.jpg'
            plt.savefig(name)

        list_im = []
        for jj in range(n):
            list_im.append('org' + str(jj) + '.jpg')
        imgs    = [PIL.Image.open(i) for i in list_im]
        imgs_comb = np.hstack((np.asarray(i) for i in imgs))
        horizontal.append(imgs_comb)
        list_im = []
        for jj in range(n):
            list_im.append('rec' + str(jj) + '.jpg')
        imgs    = [PIL.Image.open(i) for i in list_im]
        imgs_comb = np.hstack((np.asarray(i) for i in imgs))
        horizontal.append(imgs_comb)

        imgs_comb = np.vstack((np.asarray(i) for i in horizontal))
        imgs_comb = PIL.Image.fromarray(imgs_comb)
        imgs_comb.save('recon' + '.jpg')

    def reconstruction(self, vis):
        import matplotlib.pyplot as plt
        from PIL import Image
        import PIL.Image
        print('plotting original')
        plt.matshow(vis.reshape(28, 28), fignum=100, cmap=plt.cm.gray)
        plt.savefig('org.jpg')
        # encoder
        temp = np.copy(vis).reshape(784, 1)
        prob = np.dot(self.W1, temp) + self.hbias1
        prob = expit(prob)
        prob = np.dot(self.W22, prob) + self.hbias2
        prob = expit(prob)
        if self.layer == 4:  # todo bottleneck
            prob = np.dot(self.W33, prob) + self.hbias3
            prob = expit(prob)
        # decoder
            prob = np.dot(self.W33.T, prob) + self.vbias3
            prob = expit(prob)
        prob = np.dot(self.W22.T, prob) + self.vbias2
        prob = expit(prob)
        prob = np.dot(self.W1.T, prob) + self.vbias1
        prob = expit(prob)
        print('plotting reconstruction')
        plt.matshow(prob.reshape(28, 28), fignum=100, cmap=plt.cm.gray)
        plt.savefig('rec.jpg')

    def recon(self, vis):
        prob = np.dot(self.W1, vis) + self.hbias1
        prob = expit(prob)
        prob = np.dot(self.W22, prob) + self.hbias2
        prob = expit(prob)
        if self.layer == 4:  # todo bottleneck
            prob = np.dot(self.W33, prob) + self.hbias3
            prob = expit(prob)
        # decoder
            prob = np.dot(self.W33.T, prob) + self.vbias3
            prob = expit(prob)
        prob = np.dot(self.W22.T, prob) + self.vbias2
        prob = expit(prob)
        prob = np.dot(self.W1.T, prob) + self.vbias1
        prob = expit(prob)
        MSE = np.mean((prob - vis) ** 2)
        return MSE

    def scatter(self, vis, colors):
        import matplotlib.pyplot as plt
        prob = np.dot(self.W1, vis) + self.hbias1
        #prob = expit(prob)
        prob = np.dot(self.W22, prob) + self.hbias2
        #prob = expit(prob)
        if self.layer == 4:
            prob = np.dot(self.W33, prob) + self.hbias3
            #prob = expit(prob)

        print(prob.shape)
        x = prob[0, :]
        y = prob[1, :]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.scatter(x, y, c=colors, label='data')
        plt.gca().legend(loc='upper left')
        plt.show()

    def save(self, file_name):
        """ Function to save all main parameters """
        print('Saving model to: {0}'.format(file_name))
        if self.layer == 1:
            with open(file_name, 'wb') as f:
                pickle.dump(self.W, f)
                pickle.dump(self.W2, f)
                pickle.dump(self.W3, f)
                pickle.dump(self.vbias, f)
                pickle.dump(self.hbias, f)
                pickle.dump(self.dW, f)
                pickle.dump(self.dW_prev, f)
        elif self.layer == 2:
            with open(file_name, 'wb') as f:
                pickle.dump(self.W, f)
                pickle.dump(self.vbias, f)
                pickle.dump(self.hbias, f)
                pickle.dump(self.W1, f)
                pickle.dump(self.vbias1, f)
                pickle.dump(self.hbias1, f)
        elif self.layer == 3:
            with open(file_name, 'wb') as f:
                pickle.dump(self.W, f)
                pickle.dump(self.vbias, f)
                pickle.dump(self.hbias, f)
                pickle.dump(self.W1, f)
                pickle.dump(self.vbias1, f)
                pickle.dump(self.hbias1, f)
                pickle.dump(self.W22, f)  # second couplings
                pickle.dump(self.vbias2, f)
                pickle.dump(self.hbias2, f)

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
