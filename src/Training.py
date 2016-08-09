import SamplingEMF
import SamplingGibbs
import numpy as np
import copy


def calculate_weight_gradient(rbm, h_pos, v_pos, h_neg, v_neg, lr, approx="CD"):
    # Subtract step buffer from positive-phase to get gradient
    rbm.dW = lr * (np.dot(h_pos, v_pos.T))

    # Load step buffer with negative-phase
    rbm.dW -= lr*(np.dot(h_neg, v_neg.T))

    # Second-Order EMF Correction (for EMF-TAP2, EMF-TAP3)
    if ("tap2" in approx) or ("tap3" in approx):
        buf2 = rbm.W * np.dot(h_neg - h_neg ** 2, (v_neg - v_neg ** 2).T)
        rbm.dW -= lr * buf2

    # Third-Order EMF Correction (for EMF-TAP3)
    if "tap3" in approx:
        buf3 = 2 * rbm.W2 * np.dot((h_neg - h_neg ** 2) * (0.5 - h_neg), ((v_neg - v_neg ** 2) * (0.5 - v_neg)).T)
        rbm.dW -= lr * buf3

    # Gradient update on biases
    rbm.hbias += lr * (np.sum(h_pos, 1) - np.sum(h_neg, 1)).reshape(rbm.hbias.shape)  # enforce the column vector
    rbm.vbias += lr * (np.sum(v_pos, 1) - np.sum(v_neg, 1)).reshape(rbm.vbias.shape)

    # Apply Momentum (adding last gradient to this one)
    rbm.dW += rbm.momentum * rbm.dW_prev


def regularize_weight_gradient(rbm, LearnRate, L2Penalty=None, L1Penalty=None, DropOutRate=None):
    # Quadratic penalty on weights (Energy shrinkage)
    if L2Penalty is not None:
        rbm.dW -= LearnRate * L2Penalty * rbm.W

    # Linear penalty on weights (Sparsifying)
    if L1Penalty is not None:
        rbm.dW -= LearnRate * L1Penalty * np.sign(rbm.W)


def update_weights(rbm, approx):
    rbm.W += rbm.dW
    rbm.dW_prev = copy.deepcopy(rbm.dW)

    rbm.W2 = rbm.W ** 2  # Update Square
    rbm.W3 = rbm.W ** 3  # Update Cube


def get_negative_samples(rbm, vis_init, hid_init, approx, iterations, update):
    if ("naive" in approx) or ("tap2" in approx) or ("tap3" in approx):
        # v_samples, h_samples
        v_neg, h_neg = SamplingEMF.equilibrate(rbm, vis_init, hid_init, iterations=iterations, approx=approx, update=update)

    if "CD" in approx:
        #v_samples, v_means, h_samples, h_means
        v_neg, v_means, h_samples, h_neg = SamplingGibbs.MCMC(rbm, hid_init, iterations=iterations, StartMode="hidden")

    return v_neg, h_neg


def fit_batch(rbm, vis,
                    persistent=True, lr=0.1, NormalizationApproxIter=1,
                    weight_decay=None, decay_magnitude=0.01, approx="CD", update="asynch"):

    # Determine how to acquire the positive samples based upon the persistence mode.
    v_pos = copy.deepcopy(vis)
    h_samples, h_means = SamplingGibbs.sample_hiddens(rbm, v_pos)

    # Set starting points in the case of persistence
    if persistent:
        if rbm.persistent_chain_hid is None:
            # if we just initialize
            rbm.persistent_chain_hid = np.copy(h_samples)
            rbm.persistent_chain_vis = np.copy(vis)
            h_init = rbm.persistent_chain_hid
            v_init = rbm.persistent_chain_vis
        else:
            if ("naive" in approx) or ("tap2" in approx) or ("tap3" in approx):
                v_init = rbm.persistent_chain_vis
                h_init = rbm.persistent_chain_hid
            if "CD" in approx:
                v_init = np.copy(vis)
                h_init = rbm.persistent_chain_hid
    else:
        if ("naive" in approx) or ("tap2" in approx) or ("tap3" in approx):
            v_init = np.copy(vis)
            h_init = h_means
        if "CD" in approx:
            v_init = np.copy(vis)
            h_init = h_samples

    # Calculate the negative samples according to the desired approximation mode
    v_neg, h_neg = get_negative_samples(rbm, v_init, h_init, approx, NormalizationApproxIter, update)

    # If we are in persistent mode, update the chain accordingly
    if persistent:
        rbm.persistent_chain_vis = v_neg
        rbm.persistent_chain_hid = h_neg

    # Update on weights
    calculate_weight_gradient(rbm, h_means, v_pos, h_neg, v_neg, lr, approx=approx)

    if weight_decay == "l2":
        regularize_weight_gradient(rbm, lr, L2Penalty=decay_magnitude)
    elif weight_decay == "l1":
        regularize_weight_gradient(rbm, lr, L1Penalty=decay_magnitude)

    update_weights(rbm, approx)

    return rbm


def fit(rbm, data, ValidSet, lr=0.001, n_epochs=10, batch_size=100, NormalizationApproxIter=1,
             weight_decay="none", decay_magnitude=0.01, approx="CD", update="asynch",
             persistent_start=3, trace_file=None, save_file=None):

    assert 0 <= data.all() <= 1
    n_features = data.shape[0]  # dimension of data
    n_samples = data.shape[1]   # number of samples
    n_hidden = rbm.W.shape[0]  # size of hidden layer

    lr /= batch_size  # can be divided here or in the gradient update
    batch_order = np.arange(int(n_samples / batch_size))

    if trace_file is not '':
        with open(trace_file, 'w') as f:
            f.write('train: LL, normLL, EMF, normEMF, MSE, normMSE, valid: LL, normLL, EMF, normEMF, MSE, normMSE,\n')
    else:
        print('train: LL, normLL, EMF, normEMF, MSE, normMSE, valid: LL, normLL, EMF, normEMF, MSE, normMSE')

    for itr in range(n_epochs):
        print('Iteration {0}'.format(itr))

        # Shuffle data to obtain proper SGD
        np.random.shuffle(data.T)

        # Check to see if we can use persistence at this epoch
        use_persistent = True if itr >= persistent_start else False

        if trace_file is not '':
            saveResults(trace_file, rbm, data, ValidSet, n_hidden, n_features, approx)
        else:
            printResults(rbm, data, ValidSet, n_hidden, n_features, approx)

        for index in batch_order:
            if index % 100 == 0:
                print('Batch {0}'.format(index))

            batch = data[:, index * batch_size: (index + 1) * batch_size]
            fit_batch(rbm, batch, persistent=use_persistent,
                      NormalizationApproxIter=NormalizationApproxIter,
                      weight_decay=weight_decay,
                      decay_magnitude=decay_magnitude,
                      lr=lr, approx=approx,
                      update=update
            )
        # if itr % 10 == 0:
        #     print('saving')
        #     # saving parameters
        #     if save_file:
        #         rbm.save(save_file + 'epochs' + str(itr))
    if save_file:
        rbm.save(save_file)

    return rbm


def saveResults(trace_file, rbm, trainSet, validSet, n_hidden, n_features, approx):
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
        vnormMSE = vmeanMSE / N

        f.write('{0:5.3f}, {1:5.3f}, {2:5.3f}, {3:5.3f}, {4:5.3f}, {5:5.3f}, {6:5.3f}, {7:5.3f}, {8:5.3f}, {9:5.3f}, {10:5.3f}, {11:5.3f}\n'.format(\
                tmeanLL, tnormLL, tEMF, tnormEMF, tmeanMSE, tnormMSE, vmeanLL, vnormLL, vEMF, vnormEMF, vmeanMSE, vnormMSE)
        )


def printResults(rbm, trainSet, validSet, n_hidden, n_features, approx):
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
    vnormMSE = vmeanMSE / N

    print('{0:5.3f}, {1:5.3f}, {2:5.3f}, {3:5.3f}, {4:5.3f}, {5:5.3f}, {6:5.3f}, {7:5.3f}, {8:5.3f}, {9:5.3f}, {10:5.3f}, {11:5.3f}\n'.format( \
        tmeanLL, tnormLL, tEMF, tnormEMF, tmeanMSE, tnormMSE, vmeanLL, vnormLL, vEMF, vnormEMF, vmeanMSE, vnormMSE)
    )



