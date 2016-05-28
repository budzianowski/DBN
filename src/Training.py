import SamplingEMF
import SamplingGibbs
import RBM
import numpy as np
import time
import copy


def calculate_weight_gradient(rbm, h_pos, v_pos, h_neg, v_neg, lr, approx="CD"):
    ## Stubtract step buffer from positive-phase to get gradient
    #gemm!('N', 'T', lr, h_pos, v_pos, -1.0, rbm.dW)         # dW <- LearnRate*<h_pos,v_pos> - dW
    rbm.dW = lr * (np.dot(h_pos, v_pos.T))

    ## Load step buffer with negative-phase
    #gemm!('N', 'T', lr, h_neg, v_neg, 0.0, rbm.dW)          # dW <- LearRate*<h_neg,v_neg>
    rbm.dW -= lr*(np.dot(h_neg, v_neg.T))

    ## Second-Order EMF Correction (for EMF-TAP2, EMF-TAP3)
    if "tap2" in approx:
        #buf2 = gemm('N', 'T', h_neg-abs2(h_neg), v_neg-abs2(v_neg)) .* rbm.W
        buf2 = rbm.W * np.dot(h_neg - h_neg ** 2, (v_neg - v_neg ** 2).T)
        #axpy!(-lr, buf2, rbm.dW)
        rbm.dW -= lr * buf2

    ## Third-Order EMF Correction (for EMF-TAP3)
    if "tap3" in approx:
        #buf3 = gemm('N','T', (h_neg-abs2(h_neg)) .* (0.5-h_neg), (v_neg-abs2(v_neg)) .* (0.5-v_neg)) .* rbm.W2
        buf3 = 2 * rbm.W2 * np.dot((h_neg - h_neg ** 2) * (0.5 - h_neg), ((v_neg - v_neg ** 2) * (0.5 - v_neg)).T)
        #axpy!(-2.0*lr, buf3, rbm.dW)
        rbm.dW -= lr * buf3

    # Gradient update on biases
    rbm.hbias += lr * (np.sum(h_pos, 1) - np.sum(h_neg, 1)).reshape(rbm.hbias.shape)  # enforce the column vector
    rbm.vbias += lr * (np.sum(v_pos, 1) - np.sum(v_neg, 1)).reshape(rbm.vbias.shape)

    ## Apply Momentum (adding last gradient to this one)
    rbm.dW += rbm.momentum * rbm.dW_prev


def regularize_weight_gradient(rbm, LearnRate, L2Penalty=None, L1Penalty=None, DropOutRate=None):
    ## Quadratic penalty on weights (Energy shrinkage)
    if L2Penalty is not None:
        #axpy!(-LearnRate*L2Penalty,rbm.W,rbm.dW)
        rbm.dW -= LearnRate * L2Penalty * rbm.W

    ## Linear penalty on weights (Sparsifying)
    if L1Penalty is not None:
        #axpy!(-LearnRate*L1Penalty, sign(rbm.W),rbm.dW)
        rbm.dW -= LearnRate * L1Penalty * np.sign(rbm.W)

    ## Dropout Regularization (restricted set of updates)
    if DropOutRate is not None:
        pass
        # Not yet implemented, so we do nothing.
        # TODO: Implement Drop-out, here.


def update_weights(rbm, approx):
    #axpy!(1.0,rbm.dW,rbm.W)             # Take step: W = W + dW
    rbm.W += rbm.dW
    #copy!(rbm.dW_prev, rbm.dW)          # Save the current step for future use
    rbm.dW_prev = rbm.dW

    if "tap2" in approx:
        rbm.W2 = rbm.W ** 2       # Update Square [for EMF-TAP2]

    if "tap3" in approx:
        rbm.W3 = rbm.W ** 3        # Update Cube   [for EMF-TAP3]


def get_negative_samples(rbm, vis_init, hid_init, approx, iterations):
    if ("naive" in approx) or ("tap2" in approx) or ("tap3" in approx):
        v_samples, h_samples = SamplingEMF.equilibrate(rbm, vis_init, hid_init, iterations=iterations, approx=approx)

    if "CD" in approx:
        # In the case of Gibbs/MCMC sampling, we will take the binary visible samples as the negative
        # visible samples, and the expectation (means) for the negative hidden samples.
        v_samples, v_means, h_samples, h_means = SamplingGibbs.MCMC(rbm, hid_init, iterations=iterations, StartMode="hidden")

    return v_samples, h_samples


def fit_batch(rbm, vis,
                    persistent=True, lr=0.1, NormalizationApproxIter=1,
                    weight_decay=None, decay_magnitude=0.01, approx="CD"):

    # Determine how to acquire the positive samples based upon the persistence mode.
    v_pos = copy.deepcopy(vis)
    h_samples, h_means = SamplingGibbs.sample_hiddens(rbm, v_pos)

    # Set starting points in the case of persistence
    if persistent:
        if rbm.persistent_chain_hid is None:
            # if we just initialize
            rbm.persistent_chain_hid = copy.deepcopy(h_samples)
            rbm.persistent_chain_vis = copy.deepcopy(vis)
            h_init = rbm.persistent_chain_hid
            v_init = rbm.persistent_chain_vis
        else:
            if ("naive" in approx) or ("tap2" in approx) or ("tap3" in approx):
                v_init = rbm.persistent_chain_vis
                h_init = rbm.persistent_chain_hid
            if "CD" in approx:
                v_init = copy.deepcopy(vis)
                h_init = rbm.persistent_chain_hid
    else:
        if ("naive" in approx) or ("tap2" in approx) or ("tap3" in approx):
            v_init = copy.deepcopy(vis)
            h_init = h_means
        if "CD" in approx:
            v_init = copy.deepcopy(vis)
            h_init = h_samples

    # Calculate the negative samples according to the desired approximation mode # TODO means samples use means for learning
    v_neg, h_neg = get_negative_samples(rbm, v_init, h_init, approx, NormalizationApproxIter)

    # If we are in persistent mode, update the chain accordingly
    if persistent:
        rbm.persistent_chain_vis = v_neg
        rbm.persistent_chain_hid = h_neg  # here needs to be sample

    # Update on weights
    calculate_weight_gradient(rbm, h_means, v_pos, h_neg, v_neg, lr, approx=approx)

    if weight_decay == "l2":
        regularize_weight_gradient(rbm, lr, L2Penalty=decay_magnitude)
    elif weight_decay == "l1":
        regularize_weight_gradient(rbm, lr, L1Penalty=decay_magnitude)

    update_weights(rbm, approx)

    return rbm


"""
    # Boltzmann.fit (training.jl)
    ## Function Call
        `fit(rbm::RBM, X::Mat{Float64}[, persistent, lr, batch_size, NormalizationApproxIter, weight_decay,
                                         decay_magnitude, validation,monitor_ever, monitor_vis,
                                         approx, persistent_start, save_params])`
    ## Description
    The core RBM training function. Learns the weights and biasings using
    either standard Contrastive Divergence (CD) or Persistent CD, depending on
    the user options.

    - *rbm:* RBM object, initialized by `RBM()`/`GRBM()`
    - *data:* Set of training vectors
    ### Optional Inputs
     - *persistent:* Whether or not to use persistent-CD [default=true]
     - *persistent_start:* At which epoch to start using the persistent chains. Only
                           applicable for the case that `persistent=true`.
                           [default=1]
     - *lr:* Learning rate [default=0.1]
     - *n_iter:* Number of training epochs [default=10]
     - *batch_size:* Minibatch size [default=100]
     - *NormalizationApproxIter:* Number of Gibbs sampling steps on the Markov Chain [default=1]
     - *weight_decay:* A string value representing the regularization to add to apply to the
                       weight magnitude during training {"none","l1","l2"}. [default="none"]
     - *decay_magnitude:* Relative importance assigned to the weight regularization. Smaller
                          values represent less regularization. Should be in range (0,1).
                          [default=0.01]
     - *validation:* An array of validation samples, e.g. a held out set of training data.
                     If passed, `fit` will also track generalization progress during training.
                     [default=empty-set]
     """


def fit(rbm, data, ValidSet, lr=0.001, n_epochs=10, batch_size=100, NormalizationApproxIter=1,
             weight_decay="none", decay_magnitude=0.01, approx="CD",
             persistent_start=3, trace_file=None):

    assert 0 <= data.all() <= 1
    n_features = data.shape[0]  # dimension of data
    n_samples = data.shape[1]   # number of samples
    n_hidden = rbm.W.shape[0]  # size of hidden layer

    lr /= batch_size
    batch_order = np.arange(int(n_samples / batch_size))
    if trace_file is not '':
        with open(trace_file, 'w') as f:
            f.write('train: LL, normLL, EMF, normEMF, MSE, normMSE, valid: LL, normLL, EMF, normEMF, MSE, normMSE,\n')
            
    for itr in range(n_epochs):
        print('Iteration {0}'.format(itr))

        # Check to see if we can use persistence at this epoch
        use_persistent = True if itr >= persistent_start else False

        if trace_file is not '':
            saveResults(trace_file, rbm, data, ValidSet, n_hidden, n_features)
        else:
            printResults(rbm, data, ValidSet, approx, n_hidden, n_features)

        for index in batch_order:
            if index % 100 == 0:
                print('Batch {0}'.format(index))

            batch = data[:, index * batch_size: (index + 1) * batch_size]
            fit_batch(rbm, batch, persistent=use_persistent,
                      NormalizationApproxIter=NormalizationApproxIter,
                      weight_decay=weight_decay,
                      decay_magnitude=decay_magnitude,
                      lr=lr, approx=approx
            )

    return rbm


def saveResults(trace_file, rbm, trainSet, validSet, n_hidden, n_features):
    with open(trace_file, 'a') as f:
        N = n_hidden + n_features
        tmeanLL = np.mean(rbm.score_samples(trainSet[:, 0:10000]))
        tnormLL = tmeanLL / N
        tEMF = np.mean(rbm.score_samples_TAP(trainSet[:, 0:10000]))
        tnormEMF = tEMF / N
        tmeanMSE = np.mean(rbm.recon_error(trainSet[:, 0:10000]))
        tnormMSE = tmeanMSE / N

        vmeanLL = np.mean(rbm.score_samples(validSet[:, 0:10000]))
        vnormLL = vmeanLL / N
        vEMF = np.mean(rbm.score_samples_TAP(validSet[:, 0:10000]))
        vnormEMF = vEMF / N
        vmeanMSE = np.mean(rbm.recon_error(validSet[:, 0:10000]))
        vnormMSE = vmeanMSE/ N

        f.write('{0:5.3f}, {1:5.3f}, {2:5.3f}, {3:5.3f}, {4:5.3f}, {5:5.3f}, {6:5.3f}, {7:5.3f}, {8:5.3f}, {9:5.3f}, {10:5.3f}, {11:5.3f}\n'.format(\
                tmeanLL, tnormLL, tEMF, tnormEMF, tmeanMSE, tnormMSE, vmeanLL, vnormLL, vEMF, vnormEMF, vmeanMSE, vnormMSE)
        )


def printResults(rbm, data, ValidSet, approx, n_hidden, n_features):
    if "CD" in approx:
        print("Pseudo LL")
        meanLL = np.mean(rbm.score_samples(ValidSet[:, 0:10000]))
        normLL = meanLL/(n_hidden + n_features)
        meanMSE = np.mean(rbm.recon_error(ValidSet[:, 0:10000]))
        normMSE = meanMSE/(n_hidden + n_features)
        print('Validation: LL {0:10.3f}, normalized LL {1:10.3f}, MSE {2:10.3f}, normalized MSE {3:10.3f}'.format(meanLL, normLL, meanMSE, normMSE))
        meanLL = np.mean(rbm.score_samples(data[:, 0:10000]))
        normLL = meanLL/(n_hidden + n_features)
        meanMSE = np.mean(rbm.recon_error(data[:, 0:10000]))
        normMSE = meanMSE/(n_hidden + n_features)
        print('Training: LL {0:10.3f}, normalized LL {1:10.3f}, MSE {2:10.3f}, normalized MSE {3:10.3f}'.format(meanLL, normLL, meanMSE, normMSE))
        print("EMF LL")
        meanLL = np.mean(rbm.score_samples_TAP(ValidSet[:, 0:10000]))
        normLL = meanLL/(n_hidden + n_features)
        meanMSE = np.mean(rbm.recon_error(ValidSet[:, 0:10000]))
        normMSE = meanMSE/(n_hidden + n_features)
        print('Validation: LL {0:10.3f}, normalized LL {1:10.3f}, MSE {2:10.3f}, normalized MSE {3:10.3f}'.format(meanLL, normLL, meanMSE, normMSE))
        meanLL = np.mean(rbm.score_samples_TAP(data[:, 0:10000]))
        normLL = meanLL/(n_hidden + n_features)
        meanMSE = np.mean(rbm.recon_error(data[:, 0:10000]))
        normMSE = meanMSE/(n_hidden + n_features)
        print('Training: LL {0:10.3f}, normalized LL {1:10.3f}, MSE {2:10.3f}, normalized MSE {3:10.3f}'.format(meanLL, normLL, meanMSE, normMSE))
    else:
        meanLL = np.mean(rbm.score_samples_TAP(ValidSet[:, 0:10000]))
        normLL = meanLL/(n_hidden + n_features)
        meanMSE = np.mean(rbm.recon_error(ValidSet[:, 0:10000]))
        normMSE = meanMSE/(n_hidden + n_features)
        print('Validation: LL {0:10.3f}, normalized LL {1:10.3f}, MSE {2:10.3f}, normalized MSE {3:10.3f}'.format(meanLL, normLL, meanMSE, normMSE))
        meanLL = np.mean(rbm.score_samples_TAP(data[:, 0:10000]))
        normLL = meanLL/(n_hidden + n_features)
        meanMSE = np.mean(rbm.recon_error(data[:, 0:10000]))
        normMSE = meanMSE/(n_hidden + n_features)
        print('Training: LL {0:10.3f}, normalized LL {1:10.3f}, MSE {2:10.3f}, normalized MSE {3:10.3f}'.format(meanLL, normLL, meanMSE, normMSE))


# def generate(rbm, vis_init, approx, SamplingIterations):
#     Nsamples = vis_init.shape[1]
#     Nhid     = rbm.hbias.shape[0]
#     h_init  = np.zeros((Nsamples, Nhid))
#
#     if ("naive" in approx) or ("tap" in approx):
#         temp, hid_mag = SamplingEMF.equilibrate(rbm,vis_init, hid_init, iterations=SamplingIterations, approx=approx)
#
#     if "CD" in approx:
#         temp, hid_mag, temp1, temp2 = SamplingGibbs.MCMC(rbm, vis_init, iterations=SamplingIterations, StartMode="visible")
#
#     samples, temp = SamplingGibbs.sample_visibles(rbm, hid_mag)
#
#     return reshape(samples, rbm.VisShape) # TODO check

