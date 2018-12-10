import numpy as np
from .mlfm import BaseMLFM, Dimensions
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.linalg import cho_solve

def _unpack_vector(x, shape):
    res = []
    ntot = 0
    for n in shape:
        res.append(x[ntot:ntot+n])
        ntot += n
    return res

def _get_gp_theta_shape(gp_list):
    """ Handler function for getting the shape of the
    free hyper parameters for the a list of Gaussian Processes objects
    """
    return [gp.kernel.theta.size for gp in gp_list]

def _var_mixer(free_vars, free_vars_shape, fixed_vars, is_fixed_vars):
    """
    Utility function to mix the free vars and the fixed vars
    """
    free_vars = _unpack_vector(free_vars, free_vars_shape)
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
    ind_ti = _unpack_vector(inv_ind, t_sizes)

    if h is None:
        return tt_uni, ind_ti

    elif isinstance(h, float) and h > 0:
        # augment the time vector so that diff is at most h
        ttc, inds_c = augment_times(tt_uni, h)
        data_inds = [inds_c[ind_i] for ind_i in ind_ti]
        return ttc, data_inds

    else:
        raise ValueError("h should be a float > 0")

def Mpower_action_wgrad(M, dM, x, power, dMdb=None):
    """
    Returns the result of M^{power}x and its gradient. 
    """
    res = x
    grad = [np.zeros(x.size).reshape(x.shape)]*len(dM)

    if dMdb is not None:
        grad_b = [np.zeros(x.size).reshape(x.shape)]*len(dMdb)
    
    for i in range(power):
        grad = [dM_i.dot(res) + M.dot(grad_i)
                for dM_i, grad_i in zip(dM, grad)]
        if dMdb is not None:

            grad_b = [dM_i.dot(res) + M.dot(grad_i)
                      for dM_i, grad_i in zip(dMdb, grad_b)]
            
        res = M.dot(res)

    if dMdb is None:
        return res, np.array(grad)
    else:
        return res, grad, grad_b

class EMFitMixin:
    """
    Handles MAP fitting of the MLFM-SA-Mixture model
    """

    def _em_fit(self, init, free_vars_shape, fixed_vars, is_fixed_vars, priors,
                max_nt=20, **kwargs):
        free_vars = init.copy()
        g, beta, mu_ivp = _var_mixer(init, free_vars_shape, fixed_vars, is_fixed_vars)
        beta = beta.reshape(self.dim.R+1, self.dim.D)
        mu_ivp = mu_ivp.reshape((len(self.x_train_), len(self._ifix), self.dim.K))

        pi = [np.ones(len(self._ifix)) / len(self._ifix) ]*len(self.x_train_)
        resp = self._get_responsibilities(pi, g, beta, mu_ivp, 1000, self._ifix)

        gcur = g.copy()
        for nt in range(max_nt):

            if nt < 5:
                is_nt_early = True
            else:
                is_nt_early = False

            import time
            
            # M-step
            t0 = time.time()
            free_vars = self._M_step(free_vars, resp, 2000,
                                     free_vars_shape, fixed_vars, is_fixed_vars,
                                     priors,
                                     is_nt_early=is_nt_early,
                                     **kwargs)
            t1 = time.time()
            print("Time taken for M step: ", t1-t0)
            pi = self._update_pi(resp)

            # E-Step
            # update responsibilites
            t0 = time.time()
            g, beta, mu_ivp = _var_mixer(free_vars, free_vars_shape, fixed_vars, is_fixed_vars)
            beta = beta.reshape(self.dim.R+1, self.dim.D)
            mu_ivp = mu_ivp.reshape((len(self.x_train_), len(self._ifix), self.dim.K))
            dg = np.linalg.norm(g - gcur)
            if dg < 1e-4:
                break
            else:
                # keep going
                gcur = g
                resp = self._get_responsibilities(pi, g, beta, mu_ivp, 2000, self._ifix)
            t1 = time.time()                
            print("Time take for E step ", t1-t0)


        return g, beta, pi, mu_ivp

    def _M_step(self, free_vars,
                responsibilites, alpha_prec,
                free_vars_shape, fixed_vars, is_fixed_vars, priors,
                is_nt_early=False, **kwargs):
        """Optimize wrt to free variables
        """
        logpsi = np.concatenate([gp.kernel.theta for gp in self.latentforces])


        # Has to be a nicer way of doing this using slices
        inds = [np.concatenate([self.data_inds[q] + k*self.dim.N
                                for k in range(self.dim.K)])
                for q in range(len(self.y_train_))]

        def objfunc(arg, free_vars_shape, fixed_vars):
            # unpack the vector
            g, vbeta, mu_ivp = _var_mixer(arg, free_vars_shape, fixed_vars, is_fixed_vars)
            mu_ivp = mu_ivp.reshape((len(self.x_train_), len(self._ifix), self.dim.K))
            beta = vbeta.reshape(self.dim.R+1, self.dim.D)
            loglik = 0.
            loglik_g_grad = 0.
            loglik_b_grad = 0.

            try:
                # sum over the mixture components
                for m, ifx in enumerate(self._ifix):
                    ll, ll_g_grad, ll_b_grad = \
                        self.log_likelihood_k(g, beta, mu_ivp[:, m, :], alpha_prec,
                                              responsibilites, m, ifx, inds,
                                              True, True)
                    loglik += ll
                    loglik_g_grad += ll_g_grad
                    loglik_b_grad += ll_b_grad

                logprior, logprior_g_grad, _ = self._prior_logpdf(g, logpsi, True)

                grad = [(loglik_g_grad + logprior_g_grad),
                        loglik_b_grad]

                # contributions from priors
                for vn, indx in zip(['beta',], [(1, vbeta),]):
                    try:
                        prior_logpdf = priors[vn]
                        ind, x = indx  # unpack the value of var name
                        vn_lp, vn_lp_grad = prior_logpdf(x, True)
                        logprior += vn_lp
                        grad[ind] += vn_lp_grad

                    except KeyError:
                        pass

                grad = np.concatenate([item for item, b in zip(grad, is_fixed_vars)
                                       if not b])
                return -(loglik + logprior), -grad
    
            except np.linalg.LinAlgError:
                return np.inf, np.zeros(arg.size)

        try:
            optimopts = kwargs['optimopts'].copy()
        except KeyError:
            optimopts = {'disp': True}

        if is_nt_early:
            # reduce the number of steps to speed up search
            # in early states
            optimopts['maxiter'] = 50
            optimopts['gtol'] = 1e-3

        res = minimize(objfunc, free_vars,
                       jac=True,
                       args=(free_vars_shape, fixed_vars),
                       method='BFGS',
                       options = optimopts)

        self._optimres = res  # store a copy of the results

        return res.x


    def _get_responsibilities(self, pi, g, beta, mu_ivp, alpha_prec, ifix):
        """ Gets the posterior responsibilities for each comp. of the mixture
        
        Returns
        -------
            r : list, [r1, ..., rm]
            Returns the responsibilities for each of the m outputs.
        """
        probs = [[]]*len(self.N_data)  # store the log probabilities for each output

        # sum over the number of components
        for i, ifx in enumerate(ifix):

            # Get the transition matrix
            K = self._K(g, beta, ifx)
            # Raise it to a power
            Lop = np.linalg.matrix_power(K, self.order)

            # the means for each output [NK, N_outputs]
            means = np.kron(mu_ivp[:, i, :],
                            np.ones(self.dim.N)).T
            means = Lop.dot(means)

            for q, ym in enumerate(self.y_train_):
                # subsample means using data inds
                m_i_q = means[:, q].reshape(self.dim.K, self.dim.N).T

                m_i_q = m_i_q[self.data_inds[q], :]
                lp_i_q = norm.logpdf(
                    ym.reshape(self.dim.K, self.N_data[q]).T,
                    loc=m_i_q,
                    scale=1/np.sqrt(alpha_prec))
                # sum over the dimesnions
                lp_i_q = lp_i_q.sum(-1)

                if probs[q] == []:
                    probs[q] = lp_i_q[:, None]
                else:
                    probs[q] = np.column_stack((probs[q], lp_i_q))

        # weight by mixture component probs.
        probs = [lp - pi_m[None, :]
                 for lp, pi_m in zip(probs, pi)]
        # subtract the maximum for exponential normalize
        probs = [p - p.max(axis=-1)[:, None] for p in probs]
        probs = [np.exp(p) / np.exp(p).sum(-1)[:, None] for p in probs]
        return probs

    def _update_pi(self, responsibilities):
        """Update the mixture probabilities
        """
        pi_mean = [np.mean(rm, axis=0) for rm in responsibilities]
        pi_mean = sum(pi_mean) / len(pi_mean)
        return [pi_mean] * len(responsibilities)


def kron_A_enT(A, n, N):
    """
    Fast calculation of the kronecker product
    of a matrix A with the size N basis vector
    transposed.
    """
    res = np.zeros((A.shape[0], A.shape[1]*N))
    res[:, n::N] = A
    return res

class BaseMLFMSA(BaseMLFM):

    is_tt_aug = False
    def __init__(self, *args, order=1, **kwargs):
        super(BaseMLFMSA, self).__init__(*args, **kwargs)

        if isinstance(order, int) and order > 0:
            self.order = order

        else:
            raise ValueError("The approximation error must be a "
                             " postive integer.")

    def _K(self, g, beta, ifix, eval_gradient=False, eval_b_gradient=False):
        """
        Return the discretisation of the integral opearator in Picard iteration.
        """
        W = self._get_weight_matrix(self.ttc, ifix)
        g = g.reshape((self.dim.R, self.dim.N)).T
        g = np.column_stack((np.ones(self.dim.N), g))
        
        # struct matrices A_r = sum_r g_rn * sum_d brd * Ld
        struct_mats = np.array([sum(brd*Ld
                                    for brd, Ld in zip(br, self.basis_mats))
                                for br in beta])
        #At = struct_mats[None, ...] * g[..., None, None]
        #At = At.sum(1)  # sum over R
        
        if eval_gradient:

            #I = np.eye(self.dim.K)
            
            K, K_grad = (0, [])
            for n in range(self.dim.N):
                #Anw = np.kron(At[n, ...], W[:, n][:, None])
                #K += kron_A_enT(Anw, n, self.dim.N)
                for r in range(self.dim.R):
                    Ar_x_wn = np.kron(struct_mats[r+1], W[:, n][:, None])
                    K_grad.append(kron_A_enT(Ar_x_wn, n, self.dim.N))

                # equivalent (should be a much faster way of doing this... its all zeros)
                #Ien = np.kron(I, np.eye(1, self.dim.N, n).T)
                #Ien = np.zeros((self.dim.N*self.dim.K, self.dim.K))
                #Ien[np.arange(n, self.dim.N*self.dim.K, self.dim.N),
                #np.arange(self.dim.K)] = 1.
                #
                #K += kron_A_enT(Ien, ifix, self.dim.N)

            if eval_b_gradient:
                # This is lazy
                dB = self._dKdB(g, beta, ifix)
                K = sum(brd * item for brd, item in zip(beta.ravel(), dB))
                for k in range(self.dim.K):
                    K[k*self.dim.N:(k+1)*self.dim.N, ifix+k*self.dim.N] += 1.

                return K, K_grad, dB

            return K, K_grad
        else:
            At = struct_mats[None, ...] * g[..., None, None]
            At = At.sum(1)  # sum over R
            eifx = np.eye(1, self.dim.N, ifix)
            K = 0.
            for n in range(self.dim.N):
                Anw = np.kron(At[n, ...], W[:, n][:, None])
                K += kron_A_enT(Anw, n, self.dim.N)
            
            I = np.eye(self.dim.K)
        
            K += sum(np.kron(I, np.kron(np.eye(1, self.dim.N, i).T, eifx))
                     for i in range(self.dim.N))
            return K

    def _dKdB(self, g, beta, ifix):
        """
        Rewriting of this function to reduce redundant computations
        """
        W = self._get_weight_matrix(self.ttc, ifix)
        

        # for computational efficiency we need to do
        # any scalar multiplications *before* any
        # kronecker products
        GW = np.stack((gn[:, None] * W[:, n][None, :]
                       for n, gn in enumerate(g)))

        dKdB = []
        for r in range(self.dim.R+1):
            for d in range(self.dim.D):
                dKdB.append(np.kron(self.basis_mats[d],
                                    GW[:, r, :].T))

        return dKdB

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


class FitMixin:
    """
    Adds utility functions for model fitting.
    """
    def fit(self, experiments, ifix, **kwargs):
        """
        Parameters
        ----------
        experiments : A list-like of tuples [(ti, Yi)]
            Each entry (ti, Yi) represents a set of time points
            and observations Yi from the model

        """
        method = 'EM'

        # make sure the model is reader for fitting by calling _setup

        times = []
        Y = []
        for E in experiments:
            times.append(E[0])
            Y.append(E[1])
        
        self._setup(times, ifix, **kwargs)
        self.y_train_ = [y.T.ravel() for y in Y]


        # parse kwargs to see if any args are being fixed
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

        if method == 'EM':
            return self._em_fit(init, free_vars_shape, fixed_vars, is_fixed_vars, priors, **kwargs)
        
    def _setup(self, times, ifix, **kwargs):
        """Prepares the model for fitting."""
        if not hasattr(self, 'ttc'):
            self._setup_times(times, **kwargs)

        self._setup_ifix(ifix)

    def _setup_ifix(self, ifix):
        # Optionally automate this in future?
        self._ifix = ifix

    def _setup_times(self, tt,
                     h=None, tt_aug=None,
                     multi_output=False):
        """ Handles storing of training times and
        augmenting the time vector
        """
        # numpy arrayify time vectors
        tt = [np.asarray(item) for item in tt]
        
        if True:# h is not None:
            # h is checked first
            ttc, data_inds = handle_time_inds(tt, h)
            self.x_train_ = tt
            self.ttc = ttc
            self.data_inds = data_inds
            self.is_tt_aug = True

        elif tt_aug is not None:
            # User has supplied the time points at which they want
            # to estimate the latent forces

            if multi_output:
                # add tt_aug to the times list
                tt = [tt_aug] + [t for t in tt]
                ttc, data_inds = handle_time_inds(tt, h)
                self.x_train_ = tt[1:]
                self.ttc = ttc
                self.data_inds = data_inds[1:]
                self.is_tt_aug = True
        else:
            # neither h or tt_aug is supplied
            self.x_train_ = tt
            self.ttc = ttc
            self.is_tt_aug = None

        # store the dimension variables
        _, K, R, D = self.dim
        self.dim = Dimensions(self.ttc.size, K, R, D)
        self.N_data = tuple(item.size for item in self.x_train_)

    def _fit_init(self, is_fixed_vars, **kwargs):
        """ Handles model initialisation.
        """
        init_strategies = {
            'g': lambda : np.zeros(self.dim.N*self.dim.R),
            'beta': lambda : np.row_stack((np.zeros(self.dim.D),
                                           np.eye(self.dim.R, self.dim.D))).ravel(),
            'mu_ivp': lambda : np.zeros((len(self.x_train_), len(self._ifix), self.dim.K)),
            }

        var_names = ['g', 'beta', 'mu_ivp']
        full_init = [kwargs.pop("".join((vn, '0')),
                                init_strategies[vn]()).ravel()
                     for vn in var_names]
        full_init_shape = [item.size for item in full_init]

        free_vars, fixed_vars = ([], [])
        for item, boolean in zip(full_init, is_fixed_vars):
            if boolean:
                fixed_vars.append(item)
            else:
                free_vars.append(item)
        free_vars_shape = [item.size for item in free_vars]

        return np.concatenate(free_vars), free_vars_shape, fixed_vars

    def _fit_kwarg_parser(self, **kwargs):
        var_names = ['g', 'beta', 'mu_ivp']
        is_fixed_vars = [kwargs.pop("".join((vn, "_is_fixed")), False)
                         for vn in var_names]
        return is_fixed_vars
            
class MLFMSAMix(EMFitMixin, FitMixin, BaseMLFMSA):
                
    """
    Extends the MLFMSA model to use mixtures
    """

    def log_likelihood_k(self,
                         g, beta,
                         mu_ivp, alpha_prec,
                         responsibilites,
                         m, ifx, inds,
                         eval_g_grad=False, eval_b_grad=False):
        """ Returns the log-likelihood of a component of the mixture
        """

        Km, Km_g_grad, Km_b_grad = self._K(g, beta, ifx,
                                           eval_g_grad, eval_g_grad)

        means = np.kron(mu_ivp,
                        np.ones(self.dim.N)).T

        # raise K to self.order and get the gradient of its action on
        # means
        L, Ldg, Ldb = Mpower_action_wgrad(Km, Km_g_grad, means, self.order, Km_b_grad)
        
        log_lik, log_lik_g_grad, log_lik_b_grad = (0, 0, 0)

        resp = [np.row_stack([r_q]*self.dim.K)
                for r_q in responsibilites]
        
        # sum over the number of outputs
        for q, y in enumerate(self.y_train_):
            inds_q = inds[q]
            Lim_q = L[inds_q, q]

            eta = Lim_q - y
            r_q_i = resp[q][:, m]        # responsibilities for the qth replicate
                                         # of the ith mixture component
            Reta = eta * r_q_i           # weight eta by the responsibilites

            # update the log-likelihood and gradient
            log_lik += -0.5 * (eta * Reta).sum() * alpha_prec
            log_lik_g_grad -= alpha_prec * np.array(
                [np.sum(Reta * dL[inds_q, q]) for dL in Ldg])
            log_lik_b_grad -= alpha_prec * np.array(
                [np.sum(Reta * dL[inds_q, q]) for dL in Ldb])

        return log_lik, log_lik_g_grad, log_lik_b_grad

    def log_likelihood(self, g, beta,
                       mu_ivp, alpha_prec, pi,
                       ifix):

        # Has to be a nicer way of doing this using slices
        inds = [np.concatenate([self.data_inds[q] + k*self.dim.N
                                for k in range(self.dim.K)])
                for q in range(len(self.y_train_))]
        
        logps = []
        for m, ifx in enumerate(ifix):

            Km = self._K(g, beta, ifx)
            means = np.kron(mu_ivp[:, m, :], np.ones(self.dim.N)).T

            L = np.linalg.matrix_power(Km, self.order)
            L = L.dot(means)

            # sum over the experiments
            logp_m = 0.

            for q, y in enumerate(self.y_train_):
                inds_q = inds[q]
                Lim_q = L[inds_q, q]
                eta = Lim_q - y

                logp_m += norm.logpdf(eta,
                                      scale=1/np.sqrt(alpha_prec)).sum()
            logps.append(logp_m + np.log(pi[m]))

        logps = np.array(logps)
        a = logps.max()

        res = a + np.log(np.exp(logps - a).sum())
        return res

    def _laplace(self, g, beta, mu_ivp, alpha_prec, pi):

        logpsi = np.concatenate([gp.kernel.theta
                                 for gp in self.latentforces])

        def foo(g):
            ll = self.log_likelihood(g, beta, mu_ivp, alpha_prec, pi, self._ifix)
            lp = self._prior_logpdf(g, logpsi)
            return -(ll + lp)

        res = minimize(foo, g, options={'maxiter' : 20})
        C = res.hess_inv
        return res.x, C
        
    def _prior_logpdf(self, g, logpsi, eval_gradient=False, **kwargs):
        veclogpsi = logpsi.copy()  # need to keep a copy of the vectorised var.
        logpsi_shape = _get_gp_theta_shape(self.latentforces)
        logpsi = _unpack_vector(logpsi, logpsi_shape)

        # reshape g
        g = g.reshape(self.dim.R, self.dim.N)
        ll = 0.

        if eval_gradient: ll_g_grad, ll_logpsi_grad = ([], [])
        
        for r in range(self.dim.R):
            gp = self.latentforces[r]
            kern = gp.kernel.clone_with_theta(logpsi[r])

            if eval_gradient:
                K, K_gradient = kern(self.ttc[:, None],
                                     eval_gradient=True)
            else:
                K = kern(self.ttc[:, None])


            K[np.diag_indices_from(K)] += gp.alpha
            L = np.linalg.cholesky(K)

            alpha = cho_solve((L, True), g[r, :])

            tmp = -.5 * g[r, :].dot(alpha)
            tmp -= np.log(np.diag(L)).sum()

            ll += -.5 * g[r, :].dot(alpha) - \
                  np.log(np.diag(L)).sum() - \
                  L.shape[0] / 2 * np.log(2 * np.pi)

            if eval_gradient:
                ll_g_grad.append(-alpha)
                
                tmp = np.outer(alpha, alpha)
                tmp -= cho_solve((L, True), np.eye(K.shape[0]))
                ll_logpsi_grad.append(.5 * np.einsum("ij,ijk", tmp, K_gradient))

        # concatenate things together
        if eval_gradient:
            ll_logpsi_grad = np.concatenate(ll_logpsi_grad)

        ## Inspect kwargs for any additionally passed priors
        try:
            logpsi_prior = kwargs['logpsi_prior']
            if eval_gradient:
                lp_psi, lp_psi_grad = logpsi_prior.logpdf(veclogpsi, eval_gradient=True)
                ll_logpsi_grad += lp_psi_grad
            else:
                lp_psi = logpsi_prior(logpsi, eval_gradient=False)
            ll += lp_psi  # add the contribution from psi prior

        except KeyError:
            # Ignore it if no logpsi prior passed.
            pass
            

        if eval_gradient:
            return ll, \
                   np.concatenate(ll_g_grad), \
                   ll_logpsi_grad

        return ll
