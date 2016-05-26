import numpy as np
from scipy.special import expit
import SamplingEMF
import RBMBase
import copy


def free_energy(rbm, vis):
    vb = np.dot(rbm.vbias.T, vis)  #sum(vis .* rbm.vbias, 1)
    Wx_b_log = np.sum(np.log(1 + np.exp(rbm.hbias + np.dot(rbm.W, vis))), axis=0)
    return - vb - Wx_b_log


def score_samples(rbm, vis):
    n_feat, n_samples = vis.shape
    vis_corrupted = copy.deepcopy(vis)
    idxs = np.random.random_integers(0, n_feat - 1, n_samples)
    for (i, j) in zip(idxs, range(n_samples)):  # corruption of particular bit in a given (j) sample
        vis_corrupted[i, j] = 1 - vis_corrupted[i, j]

    fe = free_energy(rbm, vis)
    fe_corrupted = free_energy(rbm, vis_corrupted)

    logPL = n_feat * np.log(expit(fe_corrupted - fe))

    return logPL


def recon_error(rbm, vis):
    # Fully forward MF operation to get back to visible samples
    vis_rec = RBMBase.ProbVisCondOnHid(rbm, RBMBase.ProbHidCondOnVis(rbm, vis))
    # Get the total error over the whole tested visible set,
    # here, as MSE # change the difference!

    # MSE
    # dif = vis_rec - vis
    # mse = np.sum(dif ** 2, 0)
    # Cross-Entropy
    mse = np.sum(vis * np.log(vis_rec) + (1 - vis) * np.log(1 - vis_rec), 0)

    return mse


def score_samples_TAP(rbm, vis, n_iter=5, approx="tap2"):
    m_vis, m_hid = SamplingEMF.iter_mag(rbm, vis, iterations=n_iter, approx=approx)
    eps = 1e-6
    # clipping to compute entropy
    m_vis = np.clip(m_vis, eps, 1 - eps)
    m_hid = np.clip(m_hid, eps, 1 - eps)

    m_vis2 = m_vis ** 2
    m_hid2 = m_hid ** 2

    Entropy = np.sum(m_vis * np.log(m_vis) + (1.0 - m_vis) * np.log(1.0 - m_vis), 0) \
              + np.sum(m_hid * np.log(m_hid) + (1.0 - m_hid) * np.log(1.0 - m_hid), 0)
    Naive = np.sum(rbm.vbias * m_vis, 0) + np.sum(rbm.hbias * m_hid, 0) + np.sum(m_hid * np.dot(rbm.W, m_vis), 0)
    Onsager = 0.5 * np.sum((m_hid-m_hid2) * np.dot(rbm.W2, m_vis-m_vis2), 0)

    fe_tap = Entropy - Naive - Onsager

    if "tap3" in approx:
        # TODO third term
        pass
        fe_tap += 0

    fe = free_energy(rbm, vis)

    return -fe + fe_tap
