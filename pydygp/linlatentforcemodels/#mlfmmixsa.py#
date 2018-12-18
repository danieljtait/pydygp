import autograd.numpy as np
import autograd
from . import util
from .mlfm import BaseMLFM, Dimensions
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.stats import norm
from scipy.linalg import cho_solve
"""
Utility functions for handling the time vectors.
"""
# Utility functions for padding the time vector
def augment_times(t, h):
    """
    Pads the vector t so that the maximum spacing is h.

    Parameters
    ----------

    t : one dimensional array like

    h : float
    """
    inds = [0]
    res = [t[0]]
    for ta, tb in zip(t[:-1], t[1:]):
        N = np.ceil((tb - ta) / h + 1)
        N = int(N)
        _tt = np.linspace(ta, tb, N)
        res = np.concatenate((res, _tt[1:]))
        inds.append(inds[-1] + N - 1)
    return res, np.array(inds)

def handle_time_inds(times, h=None):
    """
    Takes a list of time vectors and returns the unique, potentially augmented,
    vector 
    """
    # get size of each vector
    t_sizes = [len(t) for t in times]

    # concatenate to single time vector
    tt = np.concatenate(times)

    # get the distinct times, and the indices
    tt_uni, inv_ind = np.unique(tt, return_inverse=True)

    # split inv_ind up
    ind_ti = util._unpack_vector(inv_ind, t_sizes)

    if h is None:
        return tt_uni, ind_ti

    elif isinstance(h, float) and h > 0:
        # augment the time vector so that diff is at most h
        ttc, inds_c = augment_times(tt_uni, h)
        data_inds = [inds_c[ind_i] for ind_i in ind_ti]
        return ttc, data_inds

    else:
        raise ValueError("h should be a float > 0")

def _var_mixer(free_vars, free_vars_shape, fixed_vars, is_fixed_vars):
    """
    Utility function to mix the free vars and the fixed vars
    """
    free_vars = util._unpack_vector(free_vars, free_vars_shape)
    if is_fixed_vars is None:
        return free_vars
    else:
        full_vars = []
        ifree, ifixed = (0, 0)
        for b in is_fixed_vars:
            if b:
                full_vars.append(fixed_vars[ifixed])
                ifixed += 1
            else:
                full_vars.append(free_vars[ifree])
                ifree += 1
        return full_vars

"""
Functions for forward in integrating the mlfm 
"""
def get_next_layer(prev_layer, offset, A, Weights):
    """
    Applies the Picard iteration to the current layer.
    """
    N, _, _ = prev_layer.shape

    # precompute A.dot(prev_layer)
    Az = np.einsum('nij,njm->nim', A, prev_layer)
    new_layer = []
    for n in range(N):
        new_layer.append(np.sum(Az * Weights[n, :][:, None, None], axis=0) + \
                         offset.T)

    return np.array(new_layer)


    
class BaseMLFMSA(BaseMLFM):
    is_tt_aug = False
    def __init__(self, *args, order=1, **kwargs):
        super(BaseMLFMSA, self).__init__(*args, **kwargs)

        if isinstance(order, int) and order > 0:
            self.order = order

        else:
            raise ValueError("The approximation error must be a "
                             " postive integer.")

    def _forward(self, g, beta, initval, ifx):
        """
        Applies the forward iteration of the Picard series
        """
        g = g.reshape((self.dim.R, self.dim.N)).T

        struct_mats = np.array([sum(brd * Ld
                                    for brd, Ld in zip(br, self.basis_mats))
                                for br in beta])

        # construct the sets of A(t_n) shape: (N, K, K)
        A = np.array([sum(gnr * Ar
                          for gnr, Ar in zip(gn, struct_mats[1:])) +
                      struct_mats[0]
                      for gn in g])

        # initial layer shape (N, K, N_samples)
        layer = np.dstack([np.row_stack([m]*self.dim.N)
                           for m in initval])

        # weight matrix
        weights = self._get_weight_matrix(self.ttc, ifx)

        for m in range(self.order):
            layer = get_next_layer(layer,
                                   initval,
                                   A,
                                   weights)

        return layer

    def forward_error(self, g, beta, alpha, initval, ifx, resp):
        layer = self._forward(g, beta, initval, ifx)

        err = sum(
            np.sum( resp[i][:, None] *
                    (layer[self.data_inds[i], :, i] - y) ** 2)
            for i, y in enumerate(self.Y_train_))
        return .5 * alpha * err

    def _get_weight_matrix(self, tt, i, store=True):
        """
        Weight matrix for the trapezoidal rule.
        """
        try:
            return self._weights[i]

        except:

            def _get_weights(tt):
                tt = np.asarray(tt)
                W = np.zeros((tt.size, tt.size))
                h = np.diff(tt)
                for i in range(len(tt)):
                    W[i, :i] += .5 * h[:i]
                    W[i, 1:i+1] += .5 * h[:i]
                return W

            ttb = tt[:i+1]
            ttf = tt[i:]

            Wb = _get_weights([t for t in reversed(ttb)])
            Wb = np.flip(Wb, (0, 1))
            Wf = _get_weights(ttf)

            W = np.zeros((len(tt), len(tt)))
            W[:i+1, :i+1] += Wb
            W[i:, i:] += Wf

            if store:
                if hasattr(self, '_weights'):
                    self._weights[i] = W
                else:
                    self._weights = {i: W}

            return W

class EMFitMixin:
    """
    """
    def _em_fit(self, init, free_vars_shape, fixed_vars, is_fixed_vars,
                priors, max_nt=50, min_nt=0, gtol=1e-3, verbose=False, **kwargs):

        # Check any arguments to be passed to the M-step optimisatio function
        optim_opts = kwargs.pop('optim_opts', {})

        # unpack the inital values
        g, beta, mu_ivp = _var_mixer(init, free_vars_shape, fixed_vars, is_fixed_vars)

        alpha = 1000.

        # do some reshaping
        beta = beta.reshape((self.dim.R+1, self.dim.D))
        mu_ivp = mu_ivp.reshape((len(self._ifix),
                                 len(self.Y_train_),
                                 self.dim.K))

        # initalise pi uniformly
        pi = np.ones(len(self._ifix)) / len(self._ifix)
        
        # get the initial responsibilities
        r = self._get_responsibilities(pi, g, beta, mu_ivp, alpha)

        free_vars = init.copy()

        for nt in range(max_nt):
            free_vars = self._M_step(free_vars, r, alpha,
                                     free_vars_shape, fixed_vars, is_fixed_vars,
                                     priors,
                                     optim_opts=optim_opts,
                                     **kwargs)

            g_, beta, mu_ivp = _var_mixer(free_vars, free_vars_shape, fixed_vars, is_fixed_vars)
            beta = beta.reshape((self.dim.R+1, self.dim.D))
            mu_ivp = mu_ivp.reshape((len(self._ifix),
                                     len(self.Y_train_),
                                     self.dim.K))
            
            pi = self._update_pi(r)

            # check for convergence
            dg = np.linalg.norm(g_ - g)
            if verbose:
                print('iter {}. Delta g: {}'.format(nt+1, dg))
            if dg <= gtol and nt >= min_nt:
                break

            else:
                g = g_
            
                # E-step
                r = self._get_responsibilities(pi, g, beta, mu_ivp, alpha)

        self.g_ = g_
        self.beta_ = beta
        self.mu_ivp_ = mu_ivp

    def _M_step(self, free_vars, resp, alpha,
                free_vars_shape, fixed_vars, is_fixed_vars, priors,
                optim_opts={},
                **kwargs):

        # inconvenient reshaping of responsibilities
        responsibs = ([item[:, i] for item in resp]
                      for i in range(len(self._ifix)))

        Cg = self.latentforces[0].kernel(self.ttc[:, None])
        Cg[np.diag_indices_from(Cg)] += 1e-5
        Lg = np.linalg.cholesky(Cg)
        Cginv = cho_solve((Lg, True), np.eye(Lg.shape[0]))

        rr = [*responsibs]

        def _objfunc(arg):
            g, vbeta, mu_ivp = _var_mixer(arg, free_vars_shape, fixed_vars, is_fixed_vars)

            # some reshaping
            beta = vbeta.reshape((self.dim.R+1, self.dim.D))
            mu_ivp = mu_ivp.reshape((len(self._ifix),
                                     len(self.Y_train_),
                                     self.dim.K))
            vals = []
            for i, ifx in enumerate(self._ifix):
                vals.append(
                    self.forward_error(g, beta, alpha, mu_ivp[i], ifx, rr[i]))

            logprior = -0.5*np.dot(g, np.dot(Cginv, g))

            for vn, x in zip(['beta'], [vbeta]):
                try:
                    prior_logpdf = priors[vn]
                    logprior = logprior + prior_logpdf(x)
                except KeyError:
                    pass
            return np.sum(vals) - logprior

        res = minimize(autograd.value_and_grad(_objfunc),
                       free_vars,
                       jac=True, **optim_opts)
        return res.x
    
    def _get_responsibilities(self, pi, g, beta, mu_ivp, alpha):
        """ Gets the posterior responsibilities for each comp. of the mixture.
        """
        probs = [[]]*len(self.N_data)
        for i, ifx in enumerate(self._ifix):

            zM = self._forward(g, beta, mu_ivp[i], ifx)

            for q, yq in enumerate(self.Y_train_):
                logprob = norm.logpdf(
                    yq, zM[self.data_inds[q], :, q], scale=1/np.sqrt(alpha))

                # sum over the dimension component
                logprob = logprob.sum(-1)

                if probs[q] == []:
                    probs[q] = logprob

                else:
                    probs[q] = np.column_stack((probs[q], logprob))
        probs = [lp - pi for lp in probs]
        # subtract the maxmium for exponential normalize
        probs = [p - np.atleast_1d(p.max(axis=-1))[:, None]
                 for p in probs]
        probs = [np.exp(p) / np.exp(p).sum(-1)[:, None] for p in probs]

        return probs

    def _update_pi(self, responsibilities):
        """Update the mixture probabilities
        """
        pi_mean = [np.mean(rm, axis=0) for rm in responsibilities]
        pi_mean = sum(pi_mean) / len(pi_mean)
        return pi_mean

class FitMixin:
    """
    Adds utility functions for model fitting.
    """
    def fit(self, experiments, ifix, **kwargs):
        """
        Parameters
        ----------
        experiments : A list-like of tuples [(ti, Yi), ]
            Each entry (ti, Yi) represents a set of time points
            and corresponding (Ni, K) array of observations.
        """

        times, Y = ([], [])
        for E in experiments:
            times.append(E[0])
            Y.append(E[1])
        
        self._setup(times, ifix, **kwargs)
        self.Y_train_ = Y  # no vectorisation!

        # parse kwargs to see which variables are being kept fixed
        is_fixed_vars = self._fit_kwarg_parser(**kwargs)

        # get the initial values
        init, free_vars_shape, fixed_vars = \
              self._fit_init(is_fixed_vars, **kwargs)

        # parse kwargs for priors
        priors = {}
        for vn in ['beta']:
            key = '_'.join((vn, 'prior'))
            try:
                priors[vn] = kwargs[key].logpdf
            except KeyError:
                pass

        method = 'EM'
        if method == 'EM':
            return self._em_fit(init, free_vars_shape, fixed_vars,
                                is_fixed_vars, priors, **kwargs)

    def _setup(self, times, ifix,
               force_setup=False, **kwargs):
        """Prepares the model fot fitting."""
        
        if not hasattr(self, 'ttc') or force_setup:
            self._setup_times(times, **kwargs)

        if not hasattr(self, '_ifix') or force_setup:
            self._setup_ifix(ifix)

    def _setup_ifix(self, ifix):
        # add a default initalisation of this variable
        self._ifix = ifix 

    def _setup_times(self, tt,
                     h=None, tt_aug=None):
        """
        Handles storing of training times and augmenting the time vector.
        """
        # make sure all the times are arrays
        tt = [np.asarray(item) for item in tt]

        ttc, data_inds = handle_time_inds(tt, h)
        self.x_train_ = tt
        self.ttc = ttc
        self.data_inds = data_inds
        self.is_tt_aug = True

        # store the dimension variable
        _, K, R, D = self.dim
        self.dim = Dimensions(self.ttc.size, K, R, D)
        self.N_data = tuple(item.size for item in self.x_train_)


    def _fit_kwarg_parser(self, **kwargs):

        var_names = ['g', 'beta', 'mu_ivp']
        is_fixed_vars = [kwargs.pop(''.join((vn, '_is_fixed')), False)
                         for vn in var_names]

        return is_fixed_vars

    def _fit_init(self, is_fixed_vars, **kwargs):
        """
        Handles model initialisation.
        """
        init_strategies = {
            'g': lambda : np.zeros(self.dim.N*self.dim.R),
            'beta': lambda : np.row_stack((np.zeros(self.dim.D),
                                           np.eye(self.dim.R, self.dim.D))).ravel(),
            'mu_ivp': lambda : _mu_ivp_init()
            }

        def _mu_ivp_init():
            mu_ivp = np.zeros((len(self._ifix),
                               self.dim.K,
                               len(self.Y_train_)))

            for q, Y in enumerate(self.Y_train_):
                for k in range(self.dim.K):
                    u = interp1d(self.x_train_[q], Y[:, k])
                    ivp = u(self.ttc[self._ifix])

                    mu_ivp[:, k, q] = u(self.ttc[self._ifix])

            return mu_ivp

        var_names = ['g', 'beta', 'mu_ivp']
        full_init = [kwargs.pop(''.join((vn, '0')),
                                init_strategies[vn]()).ravel()
                     for vn in var_names]

        free_vars, fixed_vars = ([], [])
        for item, boolean in zip(full_init, is_fixed_vars):
            if boolean:
                fixed_vars.append(item)
            else:
                free_vars.append(item)
            free_vars_shape = [item.size for item in free_vars]

        return np.concatenate(free_vars), free_vars_shape, fixed_vars


class MLFMMixSA(EMFitMixin, FitMixin, BaseMLFMSA):
    """
    Mixtures of MLFM-SA models
    """

    def loglikelihood(self, g, beta, mu_ivp, alpha, pi, priors):
        
        logprobs = []
        for i, ifx in enumerate(self._ifix):
            # get the logprobability for each mixture component
            ll = 0.
            
            zM = self._forward(g, beta, mu_ivp[i], ifx)
            for q, yq in enumerate(self.Y_train_):
                ll += norm.logpdf(
                    yq, zM[..., q], scale=1/np.sqrt(alpha)).sum()

            logprobs.append(ll + np.log(pi[i]))
        logprobs = np.array(logprobs)

        lpmax = max(logprobs)

        loglik = lpmax + np.log(np.exp(logprobs - lpmax).sum())

        Cg = self.latentforces[0].kernel(self.ttc[:, None])
        Cg[np.diag_indices_from(Cg)] += 1e-5
        Lg = np.linalg.cholesky(Cg)
        logprior = -0.5 * g.dot(cho_solve((Lg, True), g)) - \
                   np.log(np.diag(Lg)).sum() - \
                   Lg.shape[0] / 2 * np.log(2 * np.pi)


        for vn, x in zip(['beta'], beta):
            try:
                prior_logpdf = priors[vn]
                logprior += prior_logpdf(x)
            except KeyError:
                pass

        return loglik + logprior
