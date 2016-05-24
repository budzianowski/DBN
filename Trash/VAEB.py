#!/usr/bin/env python
#

import theano as th
import theano.tensor as T
import numpy as np
import gzip
import time
try :
    import cPickle as pickle
except :
    import pickle
import sys
import copy
from theano import shared
from matplotlib.pyplot import plot

import VAEBImage

floatX = th.config.floatX

#   to add another command line argument, simply add:
#   its name as a key
#   value as a tuple of its default value and the argument type (e.g. int, string, float)
command_line_args = {'seed' : (15485863, int),
                     'n_latent' : (10, int),
                     'n_epochs' : (2000, int),
                     'batch_size' : (100, int),
                     'L' : (1, int),
                     'hidden_unit' : (-1, int),
                     'learning_rate' : (0.01, float),
                     'trace_file' : ('', str),          #if set, trace information will be written about number of training
                                                        #samples and lower bound
                     'save_file' : ('', str),           #if set will train model and save it to this file
                     'load_file' : ('', str),           #if set will load model from this file, and won't traina new model
                     'vb_param_file' : ('', str)}       #file where param values for full VB are stored
#to add a new flag, simply add its name
command_line_flags = ['continuous', 'generic_estimator', 'full_varational']


def reparam_trick(mu, log_sigma, srng) :
    eps = srng.normal(mu.shape)  # shared random variable, Theano magic

    # reparametrization trick
    z = mu + T.exp(0.5 * log_sigma) * eps

    return z

class VAEB(object):
    def initialize_params(self, params) :
        # Initialization of weights (notation from pg.11):
        initW = lambda dimIn, dimOut: self.prng.normal(0,  self.sigmaInit, (dimIn, dimOut)).astype(floatX)
        initB = lambda dimOut: np.zeros((dimOut, )).astype(floatX)

        # Notation as in the pg. 11
        # Encoder
        # h
        W_values = initW(self.input_size, self.n_hidden_units)
        b_values = initB(self.n_hidden_units)
        self.W3 = th.shared(value=W_values, name='W3')
        self.b3 = th.shared(value=b_values, name='b3')

        # mu
        W_values = initW(self.n_hidden_units, self.n_latent)
        b_values = initB(self.n_latent)
        self.W4 = th.shared(value=W_values, name='W4')
        self.b4 = th.shared(value=b_values, name='b4')

        # Initialization of weights (notation from pg.11):
        initW = lambda dimIn, dimOut: self.prng.normal(0,  self.sigmaInit, (dimIn, dimOut)).astype(floatX)
        initB = lambda dimOut: np.zeros((dimOut, )).astype(floatX)

        # Notation as in the pg. 11
        # Encoder
        # h
        W_values = initW(self.input_size, self.n_hidden_units)
        b_values = initB(self.n_hidden_units)
        self.W3 = th.shared(value=W_values, name='W3')
        self.b3 = th.shared(value=b_values, name='b3')

        # mu
        W_values = initW(self.n_hidden_units, self.n_latent)
        b_values = initB(self.n_latent)
        self.W4 = th.shared(value=W_values, name='W4')
        self.b4 = th.shared(value=b_values, name='b4')

        # sigma
        W_values = initW(self.n_hidden_units, self.n_latent)
        b_values = initB(self.n_latent)
        self.W5 = th.shared(value=W_values, name='W5')
        self.b5 = th.shared(value=b_values, name='b5')

        # Decoder
        # tanh layer
        W_values = initW(self.n_latent,  self.n_hidden_units)
        b_values = initB(self.n_hidden_units)
        self.W1 = th.shared(value=W_values, name='W1')
        self.b1 = th.shared(value=b_values, name='b1')

        W_values = initW(self.n_hidden_units, self.input_size)  # or mu for continuous output
        b_values = initB(self.input_size)
        self.W2 = th.shared(value=W_values, name='W2')
        self.b2 = th.shared(value=b_values, name='b2')

        if self.continuous:  # for Freyfaces
            W_values = initW(self.n_hidden_units, self.input_size)  # sigma for gaussian output
            b_values = initB(self.input_size)
            self.W6 = th.shared(value=W_values, name='W6')
            self.b6 = th.shared(value=b_values, name='b6')

            self.params = [self.W3, self.W4, self.W5, self.W1, self.W2, \
                self.W6, self.b3, self.b4, self.b5, self.b1, self.b2, self.b6]
        else:
            self.params = [self.W3, self.W4, self.W5, self.W1, self.W2, \
                self.b3, self.b4, self.b5, self.b1, self.b2]

        if self.fullVariational :
            assert params is not None
            self.params = params
            self.full_variational_params = []
            for param in params :
                mus = th.shared(value=param.eval(), name=param.name + '_mu_vb')
                values = T.ones_like(param) * self.fullVBSigmaInit
                sigmas = th.shared(value=values.eval(), name=param.name + '_sigma_vb')
                self.full_variational_params += [mus, sigmas]

    def sample_variational_params(self) :
        for i in range(len(self.params)) :
            p = self.srng.normal(self.params[i].shape) * T.sqrt(T.sqr(self.full_variational_params[2*i+1])) + self.full_variational_params[2*i]


    def __init__(self, x_train, continuous, hidden_units, latent_size, batch_size,
                 L, learning_rate, genericEstimator, fullVariational, params=None, prng=None, sigmaInit=None):

        [self.N, self.input_size] = x_train.shape  # number of observations and features
        self.n_hidden_units = hidden_units
        self.n_latent = latent_size  # size of z
        self.continuous = continuous  # if we want to use MNIST or Frey data set
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.prng = np.random.RandomState(10) if prng is None else prng
        self.sigmaInit = 0.01 if sigmaInit is None else sigmaInit   # variance to initialize parameters, from pg. 7
        self.L = L  # number of samples from p(z|x)
        self.eps = 1e-6
        self.rho = 0.95
        self.fullVBSigmaInit = 1e-3
        self.batch_size = batch_size
        self.prng = np.random.RandomState(10)
        self.sigmaInit = 0.01    # variance to initialize parameters, from pg. 7
        self.L = L  # number of samples from p(z|x)
        self.genericEstimator = genericEstimator
        self.fullVariational = fullVariational

        if self.fullVariational :
            assert params is not None

        # creating random variable for reparametrization trick
        self.srng = T.shared_randomstreams.RandomStreams(seed=10)                #TODO want another seed?

        if params is None :
            self.initialize_params(params)
        elif not self.fullVariational :
            self.params = params
            if continuous :
                [self.W3, self.W4, self.W5, self.W1, self.W2, self.W6, self.b3, self.b4, self.b5, self.b1, self.b2, self.b6] = self.params
            else :
                [self.W3, self.W4, self.W5, self.W1, self.W2, self.b3, self.b4, self.b5, self.b1, self.b2] = self.params 
        else :
            self.initialize_params(params)
            if continuous :
                [self.W3, self.W4, self.W5, self.W1, self.W2, self.W6, self.b3, self.b4, self.b5, self.b1, self.b2, self.b6] = self.params
            else :
                [self.W3, self.W4, self.W5, self.W1, self.W2, self.b3, self.b4, self.b5, self.b1, self.b2] = self.params 
            


        # ADA-GRAD parameters
        self.ADA = []
        paramsForGradient = self.full_variational_params if self.fullVariational else self.params
        for paramsForGradient in paramsForGradient:
            eps_p = np.zeros_like(paramsForGradient.get_value(borrow=True), dtype=floatX)
            self.ADA.append(th.shared(eps_p, borrow=True))

        x_train = th.shared(np.asarray(x_train, dtype=floatX), name="x_train")

        # UPDATE and VALIDATE FUNCTION
        self.update, self.validate = self.getGradient(x_train)

    def save(self, file_name) :
        print('Saving model to: {0}'.format(file_name))

        with open(file_name, 'wb') as f :
            pickle.dump(self.n_hidden_units, f)
            pickle.dump(self.n_latent, f)
            pickle.dump(self.continuous, f)
            pickle.dump(self.learning_rate, f)
            pickle.dump(self.batch_size, f)
            pickle.dump(self.prng, f)
            pickle.dump(self.sigmaInit, f)
            pickle.dump(self.L, f)

            for p in self.params :
                pickle.dump(p, f)


    @staticmethod
    def load(file_name) :
        print('Loading model form : {0}'.format(file_name))
        with open(file_name, 'rb') as f :
            n_hidden_units = pickle.load(f)
            n_latent = pickle.load(f)
            continuous = pickle.load(f)
            learning_rate = pickle.load(f)
            batch_size = pickle.load(f)
            prng = pickle.load(f)
            sigmaInit = pickle.load(f)
            L = pickle.load(f)
            genericEstimator = pickle.load(f)

            params = []
            while True :
                try :
                    p = pickle.load(f)
                    params.append(p)
                except EOFError :
                    break

            #loading data in a load func, little hacky, but hey
            if continuous :
                with open('freyfaces.pkl', 'rb') as f :
                    data = pickle.load(f)  # only 1965 observations
                    f.close()
                    x_train = data[:1500]  # about 90% of the data
                    x_valid = data[1500:]
                    data = (x_train, x_valid)
            else :
                with gzip.open('mnist.pkl.gz', 'rb') as f :
                    data = pickle.load(f)  # 50000/10000/10000 observations
                    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = data

            return VAEB(x_train, continuous, n_hidden_units, n_latent, batch_size, L, learning_rate,
                        genericEstimator, False, params, prng, sigmaInit), data
            

    def encoder(self, x):
        h = T.tanh(T.dot(x, self.W3) + self.b3)

        mu = T.dot(h, self.W4) + self.b4
        log_sigma = T.dot(h, self.W5) + self.b5

        return mu, log_sigma

    def decoder(self, z):
        h = T.tanh(T.dot(z, self.W1) + self.b1)

        if self.continuous:
            mu = T.nnet.sigmoid(T.dot(h, self.W2) + self.b2)
            log_sigma = T.dot(h, self.W6) + self.b6

            return (mu, log_sigma)

        else:
            y = T.nnet.sigmoid(T.dot(h, self.W2) + self.b2)

            return y

    def reconstruct(self, x, n_samples) :
        mu, log_sigma = self.encoder(x)
        if n_samples <= 0 :
            y = self.decoder(mu)
        else :
            #sample from posterior
            if self.continuous :
                #hack to find out size of variables
                (y_mu, y_log_sigma) = self.decoder(mu)
                (y_mu, y_log_sigma) = (T.zeros_like(y_mu), T.zeros_like(y_log_sigma))
            else :
                y = T.zeros(x.shape)
            for i in range(n_samples) :
                z = reparam_trick(mu, log_sigma, self.srng)
                if self.continuous :
                    (new_y_mu, new_y_log_sigma) = self.decoder(z)
                    y_mu = y_mu + new_y_mu
                    y_log_sigma = y_log_sigma + new_y_log_sigma
                else :
                    y = y + self.decoder(z)
            if self.continuous :
                y_mu = y_mu / n_samples
                y_log_sigma = y_log_sigma / n_samples
                y = (y_mu, y_log_sigma)
            else :
                y = (y / n_samples)
        if self.continuous :
            (y_mu, y_log_sigma) = y
            I = T.eye(y_mu.shape[0])
            cov = (T.pow(T.exp(y_log_sigma), 2)) * I
            y = np.random.multivariate_normal(y_mu.eval(), cov.eval())
        else :
            y = y.eval()
        return y

    def posterior_log_prob(self, x, y) :
        if self.continuous:
            (mu, log_sigma) = y
            # Log-likelihood for Gaussian
            logpXgivenZ = (- 0.5 * np.log(2 * np.pi) - 0.5 * log_sigma \
                              - 0.5 * (x - mu) ** 2 / T.exp(log_sigma)).sum(axis=1, keepdims=True)

        else:
            # Cross entropy
            logpXgivenZ = -T.nnet.binary_crossentropy(y, x).sum(axis=1, keepdims=True)  # pg.11 for MNIST

        return logpXgivenZ

    def getLA(self, x, mu, log_sigma):
        SGVB = 0
        for ii in range(self.L):
            # p(x|z)
            z = reparam_trick(mu, log_sigma, self.srng)
            y = self.decoder(z)
            # prior
            prior = (- 0.5 * np.log(2 * np.pi) - 0.5 * z ** 2).sum(axis=1, keepdims=True)
            # logQ
            logQ = (- 0.5 * np.log(2 * np.pi) - 0.5 * log_sigma \
                        - 0.5 * (z - mu) ** 2 / T.exp(log_sigma)).sum(axis=1, keepdims=True)

            SGVB += T.sum(self.posterior_log_prob(x, y) + prior - logQ)
        SGVB /= self.L

        return SGVB

    def getLB(self, x, mu, log_sigma):
        SGVB = 0
        for ii in range(self.L):
            # p(x|z)
            z = reparam_trick(mu, log_sigma, self.srng)
            # decoding
            y = self.decoder(z)
            logpXgivenZ = self.posterior_log_prob(x, y)
            SGVB += T.sum(logpXgivenZ)
        SGVB /= self.L
        # KL
        KL = 0.5 * T.sum(1 + log_sigma - mu ** 2 - T.exp(log_sigma), axis=1, keepdims=True)
        SGVB += T.sum(KL)

        return SGVB

    #returns elbo for variational inference over parameters
    def getFVBL(self, x, mu, log_sigma) :
        SGVB = 0
        for ii in range(self.L) :
            #self.sample_variational_params()
            # p(x|z)
            z = reparam_trick(mu, log_sigma, self.srng)
            # decoding
            y = self.decoder(z)
            logpXgivenZ = self.posterior_log_prob(x, y)
            zConditionX = 0.5 * T.sum(1 + log_sigma - T.sqr(mu) - T.exp(log_sigma), axis=1, keepdims=True)
            thetaPrior = 0
            for i in range(len(self.params)) :
                mu = self.full_variational_params[2*i]
                sigma = self.full_variational_params[2*i+1]
                thetaPrior  += 0.5 * T.sum(1 + T.log(T.pow(sigma, 2)) - T.pow(mu, 2) - T.pow(sigma, 2))
            SGVB += x.shape[0] * (T.sum(logpXgivenZ) + T.sum(zConditionX)) + T.sum(thetaPrior)

        SGVB /= self.L
        return SGVB

    # MAIN function for feed-forwarding and getting update
    def getGradient(self, x_train):
        x = T.matrix('x')   # creating Theano variable for input
        index = T.iscalar('index')  # creating Theano variable for batching

        # encoding
        mu, log_sigma = self.encoder(x)

        # SGVB = KL + p(x|z) , eq. 10 or eq.6
        if self.genericEstimator:  # LA
            SGVB = self.getLA(x, mu, log_sigma)
        elif self.fullVariational :
            SGVB = self.getFVBL(x, mu, log_sigma)
        else:  # LB
            SGVB = self.getLB(x, mu, log_sigma)

        # Apply prior to parameters here to make it inference-procedure indep.
        scale = 1.0
        train_criterion = SGVB
        if not self.fullVariational :
            for param in self.params:
                train_criterion += -0.5 * scale * T.sum(param ** 2)
        else :
            for full_variational_param in self.full_variational_params :
                train_criterion += -0.5 * scale * T.sum(full_variational_param ** 2)

        # gradients
        if not self.fullVariational :
            gradients = T.grad(train_criterion, self.params)
        else :
            gradients = T.grad(train_criterion, self.full_variational_params)


        # update of parameters
        updates = self.getUpdates(gradients)
        #updates = self.getAdaDeltaUpdates(gradients)

        
        # update function
        update = th.function(
            inputs=[index],
            outputs=SGVB / self.batch_size,
            updates=updates,
            givens={
                x: x_train[index * self.batch_size: (index + 1) * self.batch_size]
            }
        )

        # getting likelihood for validation set
        validate = th.function(
            [x],
            SGVB,
            allow_input_downcast=True
        )

        return update, validate

    def getUpdates(self, gradients):
        eps = self.eps  # fudge factor for for ADA-GRAD

        # # SGD with prior (MAP) or L2 regularisation
        # updates = [
        #     (param, param + self.learning_rate * (gradient - eps * param ** 2))
        #     for param, gradient in zip(self.params, gradients)
        # ]

        # ADA-GRAD
        paramsToUpdate = self.full_variational_params if self.fullVariational else self.params
        updates = []
        for param, gradient, ada in zip(paramsToUpdate, gradients, self.ADA):
            acc = ada + T.sqr(gradient)   # squared!

            updates.append((param, param + self.learning_rate * gradient / (T.sqrt(acc) + eps)))  # MAP
            updates.append((ada, acc))

        return updates

    # Function to use AdaDelta inference. eps and rho are the parameters from
    # the paper. eps=epsilon is for numerical stability and rho controls the
    # extent to which the parameters are smoothed between iterations.
    def getAdaDeltaUpdates(self, gradients):

      updates, rho, eps = [], self.rho, self.eps

      paramsToUpdate = self.full_variational_params if self.fullVariational else self.params
      for x, g in zip(paramsToUpdate, gradients):

        # Define the initiailisation for the algorithm parameters.
        dx_ac = shared(np.zeros(x.get_value().shape, dtype=floatX))
        g_ac  = shared(np.zeros(x.get_value().shape, dtype=floatX))

        # Define intermediate variables used to update stuff.
        g_ac_new = rho * g_ac + (1.0 - rho) * T.sqr(g)
        dx = T.sqrt(dx_ac + eps) * g / T.sqrt(g_ac_new + eps)

        # Define udpates.
        updates.append((g_ac, g_ac_new))
        updates.append((x, x + dx))
        updates.append((dx_ac, rho * dx_ac + (1.0 - rho) * T.sqr(dx)))
            
      return updates

def get_arg(arg, args, default, type_) :
    arg = '--'+arg
    if arg in args :
        index = args.index(arg)
        value = args[args.index(arg) + 1]
        del args[index]     #remove arg-name
        del args[index]     #remove value
        return type_(value)
    else :
        return default


def get_flag(flag, args) :
    flag = '--'+flag
    have_flag = flag in args
    if have_flag :
        args.remove(flag)

    return have_flag

def parse_args() :
    args = copy.deepcopy(sys.argv[1:])
    arg_dict = {}
    for (arg_name, arg_args) in command_line_args.iteritems() :
        (arg_defalut_val, arg_type) = arg_args
        arg_dict[arg_name] = get_arg(arg_name, args, arg_defalut_val, arg_type)

    for flag_name in command_line_flags :
        arg_dict[flag_name] = get_flag(flag_name, args)

    if len(args) > 0 :
        print 'Have unused args: {0}'.format(args)

    return arg_dict


def print_args(args) :
    print('Parameters used:')
    print('--------------------------------------')
    for (k, v) in args.iteritems() :
        print('\t{0}: {1}'.format(k, v))
    print('--------------------------------------')

def load_model(file_name) :
    with open(file_name, 'rb') as f :
        return pickle.load(f)

def save_model(model, file_name) :
    print('Saving model to {0}'.format(file_name))
    with open(file_name, 'wb') as f :
        pickle.dump(model, f)


def train_model(args) :
    # model specification
    np.random.seed(args['seed'])
    n_latent = args['n_latent']
    n_epochs = args['n_epochs']
    continuous = args['continuous']
    batch_size = args['batch_size']
    L = args['L']
    hidden_unit = args['hidden_unit']
    learning_rate = args['learning_rate']
    trace_file = args['trace_file']
    generic_estimator = args['generic_estimator']
    full_varational = args['full_varational']
    save_file = args['save_file']
    vb_param_file = args['vb_param_file']

    print("loading data")
    if continuous:
        if hidden_unit < 0 :
            hidden_unit = 200
        f = open('freyfaces.pkl', 'rb')
        data = pickle.load(f)  # only 1965 observations
        f.close()
        x_train = data[:1500]  # about 90% of the data
        x_valid = data[1500:]
        data = (x_train, x_valid)
    else:
        if hidden_unit < 0 :
            hidden_unit = 500
        f = gzip.open('mnist.pkl.gz', 'rb')
        data = pickle.load(f)  # 50000/10000/10000 observations
        (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = data
        f.close()

    print("creating the model")
    if full_varational :
        model, tmp = VAEB.load(vb_param_file)
        params = model.params
    else :
        params = None

    model = VAEB(x_train, continuous, hidden_unit, n_latent, batch_size, L, learning_rate, generic_estimator, full_varational, params)

    print("learning")
    if len(trace_file) > 0 :
        with open(trace_file, 'w') as f :
            f.write('num_samples,L,Lvalid\n')
    batch_order = np.arange(int(model.N / model.batch_size))  # ordering of the batches
    for epoch in range(n_epochs) :
        start = time.time()
        np.random.shuffle(batch_order)
        LB = 0.0

        for batch in batch_order:
            batch_LB = model.update(batch)
            LB += batch_LB

        LB /= len(batch_order)
        LBvalidation = model.validate(x_valid) / x_valid.shape[0]
        if len(trace_file) > 0 :
            with open(trace_file, 'a') as f :
                f.write('{0},{1},{2}\n'.format(model.N * (epoch + 1), LB, LBvalidation))

        print("Epoch %s : [Lower bound: %s, time: %s]" % (epoch, LB, time.time() - start))

        print("          [Lower bound on validation set: %s]" % LBvalidation)

        if len(trace_file) > 0 :
            with open(trace_file, 'a') as f :
                f.write('{0},{1},{2}\n'.format(model.N * (epoch + 1), LB, LBvalidation))

    if len(save_file) > 0 :
        model.save(save_file)
    
    return model, data
        

def main() :
    args = parse_args()
    print_args(args)

    if len(args['load_file']) == 0 :
        model, data = train_model(args)
    else :
        model, data = VAEB.load(args['load_file'])
        

if __name__ == '__main__':
    main()
