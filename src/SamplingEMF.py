""" Base MF definitions"""
from scipy.special import expit
import numpy as np

# Naive term
def mag_vis_naive(rbm, m_vis, m_hid):
    buf = np.dot(rbm.W.T, m_hid) + rbm.vbias
    return expit(buf)


def mag_hid_naive(rbm, m_vis, m_hid):
    buf = np.dot(rbm.W, m_vis) + rbm.hbias
    return expit(buf)


# Second order
def mag_vis_tap2(rbm, m_vis, m_hid):
    buf = np.dot(rbm.W.T, m_hid) + rbm.vbias
    second_order = (m_vis - 0.5) * np.dot(rbm.W2.T, m_hid - m_hid ** 2)
    tap = buf - second_order
    return expit(tap)


def mag_hid_tap2(rbm, m_vis, m_hid):
    buf = np.dot(rbm.W, m_vis) + rbm.hbias
    second_order = (m_hid - 0.5) * np.dot(rbm.W2, m_vis - m_vis ** 2)
    tap = buf - second_order
    return expit(tap)


# Third order
def mag_vis_tap3(rbm, m_vis, m_hid):
    buf = np.dot(rbm.W.T, m_hid) + rbm.vbias
    second_order = (m_vis - 0.5) * rbm.W2.T.dot(m_hid - m_hid ** 2)
    hidden = (m_hid - m_hid ** 2)*(0.5 - m_hid)
    visible = m_vis - 3.0 * (m_vis ** 2) + 2 * (m_vis ** 3)
    third_order = visible * rbm.W3.T.dot(hidden)
    tap = buf - second_order + (1.0 / 3.0) * third_order
    return expit(tap)


def mag_hid_tap3(rbm, m_vis, m_hid):
    buf = np.dot(rbm.W, m_vis) + rbm.hbias
    second_order = (m_hid - 0.5) * rbm.W2.dot(m_vis - m_vis ** 2)
    visible = (m_vis - m_vis ** 2)*(0.5 - m_vis)
    hidden = m_hid - 3.0 * (m_hid ** 2) + 2 * (m_hid ** 3)
    third_order = hidden * rbm.W3.dot(visible)
    tap = buf - second_order + (1.0 / 3.0) * third_order
    return expit(tap)


def equilibrate(rbm, vis_init, hid_init, iterations=3, approx="tap2", damp=0.5, update="asynch"):
    m_vis = vis_init
    m_hid = hid_init
    if approx == "naive":
        mag_vis = mag_vis_naive
        mag_hid = mag_hid_naive
    elif approx == "tap2":
        mag_vis = mag_vis_tap2
        mag_hid = mag_hid_tap2
    elif approx == "tap3":
        mag_vis = mag_vis_tap3
        mag_hid = mag_hid_tap3

    # Set the proper iteration based on the approximation type
    # Take the desired number of steps
    if update == "asynch":
        for ii in range(iterations):
            m_vis = damp * mag_vis(rbm, m_vis, m_hid) + (1 - damp) * m_vis
            m_hid = damp * mag_hid(rbm, m_vis, m_hid) + (1 - damp) * m_hid
    elif update == "sequential":
        for ii in range(iterations):
            for jj in range(len(m_hid)):
                m_vis = damp * mag_vis(rbm, m_vis, m_hid) + (1 - damp) * m_vis
                m_hid[jj] = damp * mag_hid(rbm, m_vis, m_hid)[jj] + (1 - damp) * m_hid[jj]
    elif update == "parallel":
        for ii in range(iterations):
            visOld = np.copy(m_vis)
            m_vis = damp * mag_vis(rbm, m_vis, m_hid) + (1 - damp) * m_vis
            m_hid = damp * mag_hid(rbm, visOld, m_hid) + (1 - damp) * m_hid

    return m_vis, m_hid


def iter_mag(rbm, vis, iterations=5, approx="tap2", damp=0.5):
    m_vis = vis
    m_hid = rbm.probHidCondOnVis(m_vis)

    if approx == "naive":
        mag_vis = mag_vis_naive
        mag_hid = mag_hid_naive
    elif approx == "tap2":
        mag_vis = mag_vis_tap2
        mag_hid = mag_hid_tap2
    elif approx == "tap3":
        mag_vis = mag_vis_tap3
        mag_hid = mag_hid_tap3

    for ii in range(iterations):
        m_vis = damp * mag_vis(rbm, m_vis, m_hid) + (1 - damp) * m_vis
        m_hid = damp * mag_hid(rbm, m_vis, m_hid) + (1 - damp) * m_hid

    return m_vis, m_hid

