### Base MF definitions
#### Naive mean field
from scipy.special import expit
import numpy as np
from RBMBase import *


def mag_vis_naive(rbm, m_hid):
    buf = np.dot(rbm.W.T, m_hid) + rbm.vbias
    return expit(buf)


def mag_hid_naive(rbm, m_vis):
    buf = np.dot(rbm.W, m_vis) + rbm.hbias
    return expit(buf)


#### Second order development
def mag_vis_tap2(rbm, m_vis, m_hid): ## to be constrained to being only Bernoulli
    buf = np.dot(rbm.W.T, m_hid) + rbm.vbias
    second_order = (m_vis - 0.5) * rbm.W2.T.dot(m_hid - m_hid ** 2)
    tap = buf - second_order
    return expit(tap)


def mag_hid_tap2(rbm, m_vis, m_hid):
    buf = np.dot(rbm.W, m_vis) + rbm.hbias
    second_order = (m_hid - 0.5) * rbm.W2.dot(m_vis - m_vis ** 2)
    tap = buf - second_order
    return expit(tap)


#### Third order development
def mag_vis_tap3(rbm, m_vis, m_hid): ## to be constrained to being only Bernoulli
    return m_vis
    pass
    # buf = np.dot(rbm.W.T, m_hid) + rbm.vbias  # TODO: check get a vector
    # second_order = gemm('T', 'N', rbm.W2, m_hid-abs2(m_hid)).*(0.5-m_vis)
    # third_order = gemm('T', 'N', rbm.W3, abs2(m_hid).*(1.-m_hid)).*(1/3-2*(m_vis-abs2(m_vis)))
    # axpy!(1.0, second_order, buf)
    # axpy!(1.0, third_order, buf)
    # return expit(buf)


def mag_hid_tap3(rbm, m_vis, m_hid):
    return m_hid
    pass
    # buf = gemm('N', 'N', rbm.W, m_vis) .+ rbm.hbias
    # second_order = gemm('N', 'N', rbm.W2, m_vis-abs2(m_vis)).*(0.5-m_hid)
    # third_order = gemm('N', 'N', rbm.W3, abs2(m_vis).*(1.-m_vis)).*(1/3-2*(m_hid-abs2(m_hid)))
    # axpy!(1.0, second_order, buf)
    # axpy!(1.0, third_order, buf)
    # return expit(buf)


def equilibrate(rbm, vis_init, hid_init, iterations=3, approx="tap2", damp=0.5):
    # Redefine names for clarity
    m_vis = vis_init
    m_hid = hid_init

    # Set the proper iteration based on the approximation type
    # Take the desired number of steps
    if approx == "naive":
        for ii in range(iterations):
           m_vis = damp * mag_vis_naive(rbm, m_hid) + (1 - damp) * m_vis
           m_hid = damp * mag_hid_naive(rbm, m_hid) + (1 - damp) * m_hid
    elif approx == "tap3":
        for ii in range(iterations):
           m_vis = damp * mag_vis_tap3(rbm, m_vis, m_hid) + (1 - damp) * m_vis
           m_hid = damp * mag_hid_tap3(rbm, m_vis, m_hid) + (1 - damp) * m_hid
    else:
        for ii in range(iterations):
           m_vis = damp * mag_vis_tap2(rbm, m_vis, m_hid) + (1 - damp) * m_vis
           m_hid = damp * mag_hid_tap2(rbm, m_vis, m_hid) + (1 - damp) * m_hid

    return m_vis, m_hid


def iter_mag(rbm, vis, iterations=3, approx="tap2", damp=0.5):
    m_vis = vis
    m_hid = ProbHidCondOnVis(rbm, m_vis)

    if approx == "naive":
        for ii in range(iterations):
            m_vis = damp * mag_vis_naive(rbm, m_hid) + (1 - damp) * m_vis
            m_hid = damp * mag_hid_naive(rbm, m_vis) + (1 - damp) * m_hid
    elif approx == "tap3":
        for ii in range(iterations):
            m_vis = damp * mag_vis_tap3(rbm, m_vis, m_hid) + (1 - damp) * m_vis
            m_hid = damp * mag_hid_tap3(rbm, m_vis, m_hid) + (1 - damp) * m_hid
    else:
        for ii in range(iterations):
            m_vis = damp * mag_vis_tap2(rbm, m_vis, m_hid) + (1 - damp) * m_vis
            m_hid = damp * mag_hid_tap2(rbm, m_vis, m_hid) + (1 - damp) * m_hid

    return m_vis, m_hid

# def iter_mag_persist!(rbm, vis, n_times=3, approx="tap2", damp=0.5)
#     v_pos = vis
#     h_pos = ProbHidCondOnVis(rbm, v_pos)
#
#     if approx == "naive":
#         mag_vis = mag_vis_naive
#         mag_hid = mag_hid_naive
#     elif approx == "tap3":
#         mag_vis = mag_vis_tap3
#         mag_hid = mag_hid_tap3
#     else:
#         mag_vis = mag_vis_tap2
#         mag_hid = mag_hid_tap2
#
#     m_vis = rbm.persistent_chain_vis
#     m_hid = rbm.persistent_chain_hid
#
    # for ii in range(1, iterations + 1):
    #     m_vis = damp * mag_vis(rbm, m_vis, m_hid) + (1 - damp) * m_vis
    #     m_hid = damp * mag_hid(rbm, m_vis, m_hid) + (1 - damp) * m_hid
#
#     rbm.persistent_chain_vis = m_vis
#     rbm.persistent_chain_hid = m_hid
#
#     return v_pos, h_pos, m_vis, m_hid
