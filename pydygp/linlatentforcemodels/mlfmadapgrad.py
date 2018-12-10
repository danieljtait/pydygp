"""
Class definition for the mlfm with adaptive gradient matching

.. currentmodule:L pydygp.linlatentforcemodels

"""
import numpy as np
import warnings
from scipy.stats import multivariate_normal
from .mlfm import Dimensions, BaseMLFM
from scipy.linalg import (block_diag, cho_solve, cholesky, solve_triangular)
from pydygp.gradientkernels import ConstantKernel, RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.optimize import minimize
from collections import namedtuple

class MLFMAdapGradFitResults(namedtuple('MLFMAdapGradFitResults',
                                        ('g', 'beta', 'logpsi', 'logphi',
                                         'loggamma', 'logtau', 'optimres'))):
    """
    Results of MLFMAdapGrad MAP fit stored as a named tuple
    """
    def __new__(cls, g, beta, logpsi, logphi, loggamma, logtau, optimres):
        return super(MLFMAdapGradFitResults, cls).__new__(
            cls, g, beta, logpsi, logphi, loggamma, logtau, optimres)


def _unpack_vector(x, xk_shape):
    """
    Unpacks a flat vector into the len(xk_shape) subvectors
    of size xk_shape[i], i=1,...,len(xk_shape)
    """
    ntot = 0
    res = []
    for nk in xk_shape:
        res.append(x[ntot:ntot + nk])
        ntot += nk
    return res

def _get_gp_theta_shape(gp_list):
    """ Handler function for getting the shape of the
    free hyper parameters for the a list of Gaussian Processes objects
    """
    return [gp.kernel.theta.size for gp in gp_list]

def _ls_covar_k_wgrad(logpsi, gamma, tt, gradientgp,
                      return_Cxx_inv_grad=False):
    """Handler function for constructing the covariance
    matrices for the GradientKernel latent states

    Returns
    -------

    L : array_like, shape tt.size, tt.size
        Cholesky decomposition of the kernel

    Mdx : array_like
       Matrix such that Mdx.dot(x) is equivalent to E[dxdt | x]

    Schol : array_like
       Cholesky decomposition of S = Cov[dxdt | x] + gamma*I
    """
    kernel = gradientgp.kernel.clone_with_theta(logpsi)
    Cxx, Cxx_grad = kernel(tt[:, None], eval_gradient=True)
    Cxx[np.diag_indices_from(Cxx)] += gradientgp.alpha

    Cxdx, Cxdx_grad = kernel(tt[:, None], comp='xdx', eval_gradient=True)
    Cdxdx, Cdxdx_grad = kernel(tt[:, None], comp='dxdx', eval_gradient=True)
    Lxx = np.linalg.cholesky(Cxx)
    Mdx = Cxdx[..., 0].T.dot(cho_solve((Lxx, True), np.eye(tt.size)))
    S = Cdxdx[..., 0, 0] - \
        Cxdx[..., 0].T.dot(cho_solve((Lxx, True), Cxdx[..., 0]))
    S[np.diag_indices_from(S)] += gamma

    # Calculate the gradients
    P = Cxx_grad.shape[-1]  # size of parameter vector.
    Cxx_inv_grad = -np.dstack([cho_solve((Lxx, True),
                                         cho_solve((Lxx, True), Cxx_grad[:, :, p]).T)
                               for p in range(P)])
    M_grad = np.dstack([cho_solve((Lxx, True), Cxdx_grad[..., 0, p]).T
                        for p in range(P)])

    M_grad -= np.dstack([Cxdx[:, :, 0].dot(Cxx_inv_grad[:, :, p])
                         for p in range(P)])
    
    Cdx_x_grad = Cdxdx_grad[:, :, 0, 0, :].copy()
    expr = np.dstack([Cxdx_grad[:, :, 0, p].T.dot(
        cho_solve((Lxx, True), Cxdx[..., 0]))
                      for p in range(P)])    

    Cdx_x_grad -= expr
    Cdx_x_grad -= np.dstack([expr[:, :, p].T for p in range(P)])
    Cdx_x_grad -= np.dstack([Cxdx[:, :, 0].T.dot( \
            Cxx_inv_grad[:, :, p].dot( \
            Cxdx[:, :, 0])) for p in range(P)])

    if return_Cxx_inv_grad:
        return (Lxx, Mdx, np.linalg.cholesky(S)), \
               (Cxx_inv_grad, M_grad, Cdx_x_grad)
    else:
        return (Lxx, Mdx, np.linalg.cholesky(S)), \
               (Cxx_grad, M_grad, Cdx_x_grad)        

def _check_Y(Y, mlfm):
    """ Backwards compatibility for nonvectorised Y.
    """
    if len(Y.shape) == 1:
        # Y has been vectorised
        return Y
    elif len(Y.shape) == 2:
        ncol = Y.shape[1]
        nrow = Y.shape[0]
        if ncol == mlfm.dim.K and \
              nrow == mlfm._Ndata:
              # only one output, return vectorised Y
              return Y.T.ravel()
        elif nrow == mlfm._Ndata*mlfm.dim.K:
            # already columns of vectorised outputs
            return Y
        else:
            raise ValueError("Bad shape for data Y.")
    else:
        msg = "Shape of Y is not compatible."
        raise ValueError(msg)

def _sklearn_gp_falsefit(gp, y_train, X_train, kernel):
    gp.X_train_ = X_train.copy()
    gp.y_train_ = y_train.copy()
    gp.kernel_ = kernel
    K = gp.kernel_(gp.X_train_)
    K[np.diag_indices_from(K)] += gp.alpha
    gp.L_ = np.linalg.cholesky(K)
    gp.alpha_ = cho_solve((gp.L_, True), gp.y_train_)
    gp._y_train_mean = np.zeros(1)

class MLFMAdapGrad(BaseMLFM):
    """
    Multiplicative latent force model (MLFM) using the adaptive
    gradient matching approximation.

    Parameters
    ----------

    struct_mats : list of square ndarray
        A set of [A0,...,AR] square numpy array_like

    lf_kernels : list of kernel objects, optional
        Kernels of the latent force Gaussian process objects. If None
        is passed, the kernel \"1.0 * RBF(1.0)\" is used as a defualt.

    Read more in the :ref:`Tutorial <tutorials-mlfm>`.
    """
    def __init__(self,
                 basis_mats,
                 **kwargs):

        super(MLFMAdapGrad, self).__init__(basis_mats, **kwargs)

        # flags for fitting function
        self.is_tt_aug = False     # Has the time vector been augmented

    def _setup_times(self, tt, tt_aug=None, data_inds=None, **kwargs):
        """ Sets up the model time vector, optionally augments the time
        vector with additional points for prediction

        Parameters
        ----------

        tt : array_like, shape(N_data, )
            vector of times of observed data

        tt_aug : array_like, optional
            augmented vector of times with tt a subset of tt_aug

        data_inds : list, option
            indices such that tt_aug[data_inds] = tt_aug
        """
        if tt_aug is not None and data_inds is None:
            msg = "Must supply a list of integers, data_inds, " + \
                  "such that tt_aug[data_inds] == tt"
            raise ValueError(msg)

        elif tt_aug is not None and data_inds is not None:

            # check that data_inds seems to point to the right place
            if np.linalg.norm(tt - tt_aug[data_inds]) < 1e-8:
                # complete time vector
                self.ttc = tt_aug
                self.data_inds = data_inds
                self.dim = Dimensions(tt_aug.size,
                                      self.dim.K, self.dim.R, self.dim.D)

                # flag the presence of an augmented time vector
                self.is_tt_aug = True

            else:
                msg = "tt_aug[data_inds] != tt"
                raise ValueError("msg")

        else:
            self.dim = Dimensions(tt.size, self.dim.K, self.dim.R, self.dim.D)
            self.ttc = tt

    def _setup_latentstates(self, kernels=None, logphi_has_prior=False):
        """ Handles setting up of the latent state GPs.

        Parameters
        ----------

        kernels : list, optional
            list of gradientkernel objects
        """
        self.logphi_has_prior = logphi_has_prior
        
        if kernels is None:
            # Default is for kernels = 1.*exp(-.5*(s-t)**2)
            ls_kernels = [ConstantKernel(1.)*RBF(1.)
                          for k in range(self.dim.K)]
            self.latentstates = [GaussianProcessRegressor(kern) for kern in ls_kernels]

            if self.logphi_has_prior:
                from pydygp.probabilitydistributions import (ExpGeneralisedInvGaussian,
                                                             )
                # Default is ExpGeneralisedInvGaussian for the length scale
            
        else:
            raise NotImplementedError("User supplied kernels not currently supported.")


    def _setup(self, times, **kwargs):
        """ Prepares the model for fitting.
        """
        if not hasattr(self, 'ttc'):
            # make sure time and possibly aug. times have been setup
            self._setup_times(times, **kwargs)
        
        if not hasattr(self, 'latentstates'):
            # no latent states supplied so use default setting
            self._setup_latentstates()


    def fit(self, times, Y, **kwargs):
        """ Fit the MLFM using Adaptive Gradient Matching by maximimising
        the likelihood function.

        Parameters
        ----------
        times : ndarray, shape (ndata, )
            Data time points. Ordered array of size (ndata, ),
            where 'ndata' is the number of data points

        Y : ndarray, shape = (ndata, K)
            Data to be fitted. Array of size (ndata, K),
            where 'ndata' is the number of data points and K
            is the dimension of the latent variable space.

        Returns
        -------
        res_par : A :class:`.MLFMAdapGradFitResults`
        
        """
        # make sure the model is ready for fitting by calling _setup
        self._setup(times, **kwargs)

        # parse kwargs to see if any args kept fixed
        is_fixed_vars = _fit_kwarg_parser(self, **kwargs)

        y_train = _check_Y(Y, self)

        # Check for kwarg priors
        priors = {}
        for vn in ['logpsi', 'beta', 'logtau']:
            key = '_'.join((vn, 'prior'))
            try:
                priors[vn] = kwargs[key].logpdf
            except KeyError:
                pass

        # optionally fix some of the entries of beta
        beta_fix_inds = kwargs.pop('beta_fix_inds', None)

        # obj function is given by neg. log likelihood + log prior
        def objfunc(arg, free_vars_shape, fixed_vars, fixed_beta=None):
            g, vbeta, logpsi, logphi, loggamma, logtau = \
               _var_mixer(arg, free_vars_shape, fixed_vars, is_fixed_vars)
            # reshape beta
            if beta_fix_inds is None:
                beta = vbeta.reshape(self.dim.R+1, self.dim.D)
            else:
                beta = np.zeros((self.dim.R+1, self.dim.D))
                beta[np.logical_not(beta_fix_inds)] = vbeta
                beta[beta_fix_inds] = fixed_beta
                
            try:
                ll, ll_g_grad, ll_beta_grad, ll_lphi_grad, ll_lgam_grad, ll_ltau_grad = \
                    self.log_likelihood(y_train, g, beta,
                                        logphi, loggamma, logtau,
                                        eval_gradient=True)

                if beta_fix_inds is not None:
                    ll_beta_grad = \
                                 ll_beta_grad[np.logical_not(beta_fix_inds.ravel())]

                lp, lp_g_grad, lp_lpsi_grad = \
                    self._prior_logpdf(g, logpsi,
                                       eval_gradient=True,
                                       **kwargs)  # pass kwargs to logprior

                grad = [-(ll_g_grad + lp_g_grad),
                        -ll_beta_grad,
                        -lp_lpsi_grad,
                        -ll_lphi_grad,
                        -ll_lgam_grad,
                        -ll_ltau_grad]

                # possible logtau prior
                # add contributions from the various priors
                for vn, indx in zip(['beta', 'logpsi', 'logtau'],
                                    [(1, vbeta),
                                     (2, logpsi),
                                     (5, logtau)]):
                    try:
                        prior_logpdf = priors[vn]
                        ind, x = indx  # unpack the value of var. name
                        vn_lp, vn_lp_grad = prior_logpdf(x, True)
                        lp += vn_lp
    
                        grad[ind] += -vn_lp_grad

                    except KeyError:
                        # no prior specified for this var.
                        pass

                grad = np.concatenate([item for item, b in zip(grad, is_fixed_vars)
                                       if not b])
                return -(ll + lp), grad

            except np.linalg.LinAlgError:
                return np.inf, np.zeros(arg.size)

        init, free_vars_shape, fixed_vars = \
              self._fit_init(is_fixed_vars, **kwargs)
        # Handle the possibility that some of the indices of beta
        # have been kept fixed
        if beta_fix_inds is not None:
            beta_size = (self.dim.R+1)*self.dim.D
            beta_init = init[self.dim.N*self.dim.R:
                             self.dim.N*self.dim.R + beta_size]
            beta_free = beta_init[np.logical_not(beta_fix_inds.ravel())]
            fixed_beta = beta_init[beta_fix_inds.ravel()]
            init = np.concatenate((init[:self.dim.N*self.dim.R],
                                   beta_free,
                                   init[self.dim.N*self.dim.R + beta_size:]))
            free_vars_shape[1] = len(beta_free)
        else:
            fixed_beta = None

        res = minimize(objfunc, init,
                       jac=True,
                       args=(free_vars_shape, fixed_vars, fixed_beta),
                       options=kwargs.pop('optim_options', None))

        #self._optim_res = res
        
        g_, vbeta_, logpsi_, logphi_, loggamma_, logtau_ = \
            _var_mixer(res.x, free_vars_shape, fixed_vars, is_fixed_vars)

        if beta_fix_inds is None:
            beta_ = vbeta_.reshape(self.dim.R+1, self.dim.D)            
        else:
            beta_ = np.zeros((self.dim.R+1, self.dim.D))
            beta_[np.logical_not(beta_fix_inds)] = vbeta_
            beta_[beta_fix_inds] = fixed_beta
        


        # Add a kernel_ to the gps to allow them to be used
        # for prediction
        _logpsi_shape = _get_gp_theta_shape(self.latentforces)
        reshape_logpsi = _unpack_vector(logpsi_, _logpsi_shape)
        for r, gr in enumerate(g_.reshape(self.dim.R, self.dim.N)):
            gp = self.latentforces[r]
            gp.kernel_ = gp.kernel.clone_with_theta(reshape_logpsi[r])
            kern = gp.kernel.clone_with_theta(reshape_logpsi[r])
            _sklearn_gp_falsefit(gp, gr, self.ttc[:, None], kern)

        # store some variables for later use
        self.X_train_ = self.ttc[:, None]
        if not is_fixed_vars[0]:
            # Optimised g - so get laplace approx to from optim
            g_laplace_cov = res.hess_inv[:self.dim.N*self.dim.R,
                                         :self.dim.N*self.dim.R]
            g_laplace_cov = g_laplace_cov.reshape(self.dim.R,
                                                  self.dim.R,
                                                  self.dim.N,
                                                  self.dim.N)
            self.g_laplace_cov = g_laplace_cov
            #Cff = block_diag(*(gp.kernel_(self.X_train_)
            #                   for gp in self.latentforces))
            #Cff[np.diag_indices_from(Cff)] += 1e-4
            #self.g_L_ = cholesky(Cff, lower=True)
            #self.g_alpha_ = cho_solve((self.g_L_, True), g_)

        return MLFMAdapGradFitResults(g_.reshape(self.dim.R, self.dim.N),
                                      beta_, logpsi_, logphi_, loggamma_, logtau_, res)


    def predict_lf(self, tt, return_std=False):
        """
        Predict the latent force variables using the Laplace approximation.

        Requires that the model has been fitted.

        Parameters
        ----------
        tt : array-like, shape = (n, )
            Time points where the GPs are evaluated

        return_std: bool, default: False
            If True, the standard deviation of the predictive distribution at
            the query times is returned, along with the mean.

        Returns
        -------
        lf_mean : array, shape = (R, tt.size)
            Mean of predictive distribution at tt

        lf_std : array, shape = (R, tt.size), optional
            Standard deviation of the predictive distribution at tt. Only
            returned when return_std is True
        """

        Cff_ = [gp.kernel_(tt[:, None], self.X_train_)
                for gp in self.latentforces]
        lf_mean = np.concatenate([Cff_[r].dot(gp.alpha_) for
                                  r, gp in enumerate(self.latentforces)])
        lf_mean = lf_mean.reshape(self.dim.R, tt.size)

        if return_std:
            # predict the variance components
            L_inv = [solve_triangular(gp.L_.T, np.eye(gp.L_.shape[0]))
                     for gp in self.latentforces]
            K_inv = [Lr_inv.dot(Lr_inv.T) for Lr_inv in L_inv]
            y_var = np.concatenate([gp.kernel_.diag(tt[:, None])
                                    for gp in self.latentforces])
            y_var += np.concatenate(
                [np.einsum('ij,ij->i',
                           np.dot(K_trans,
                                  self.g_laplace_cov[r, r, ...] - K_inv[r]),
                           K_trans)
                for r, K_trans in enumerate(Cff_)])

            # the var. check from sklearn GaussianProcessRegressor
            # Check if any of the variances is negative because of
            # numerical issues. If yes: set the variance to 0.
            y_var_negative = y_var < 0
            if np.any(y_var_negative):
                warnings.warn("Predicted variances smaller than 0. "
                              "Setting those variances to 0.")
                y_var[y_var_negative] = 0.0

            # reshape the mean and var
            lf_mean = lf_mean.reshape(self.dim.R, tt.size)
            lf_sd = np.sqrt(y_var).reshape(self.dim.R, tt.size)
            
            return lf_mean, lf_sd

        """
        Cff_ = block_diag(*Cff_)        
        A = Cff_.dot(cho_solve((self.g_L_, True), np.eye(self.dim.N*self.dim.R)))

        K = A.dot(self.g_laplace_cov.dot(A.T))
        Cp = []
        for gp in self.latentforces:
            C22 = gp.kernel_(tt[:, None])
            C21 = gp.kernel_(tt[:, None], self.X_train_)
            Cp.append(C22 - C21.dot(cho_solve((gp.L_, True), C21.T)))
        K += block_diag(*Cp)
        """
        return lf_mean.reshape(self.dim.R, tt.size)#, np.sqrt(np.diag(K))
        

    def _fit_kwarg_parser(self, **kwargs):
        return _fit_kwarg_parser(self, **kwargs)

    def _fit_init(self, is_fixed_vars, **kwargs):
        # Initalise the value of latent force GPs
        try:
            g0 = kwargs['g0']
        except KeyError:
            g0 = np.zeros(self.dim.R*self.dim.N)
        # Initalise the hyperparameters for the latent force GPs
        try:
            logpsi0 = kwargs['logpsi0']
        except:
            logpsi0 = np.concatenate([gp.kernel.theta
                                      for gp in self.latentforces])
        # Initalise the hyperparameters for the state state GPs
        try:
            logphi0 = kwargs['logphi0']
        except:
            logphi0 = np.concatenate([gp.kernel.theta
                                      for gp in self.latentstates])
        # Initalise the gradient matching error variance
        try:
            loggamma0 = kwargs['loggamma0']
        except:
            loggamma0 = np.log(1e-4*np.ones(self.dim.K))
        # Initalise the data precisions
        try:
            logtau0 = kwargs['logtau0']
        except:
            logtau0 = np.log(1e4*np.ones(self.dim.K))
        # Initalise beta
        try:
            beta0 = kwargs['beta0']
        except:
            beta0 = np.eye(N=self.dim.R+1, M=self.dim.D)

        vbeta0 = beta0.ravel()  # vec. beta as [br1,...,brD]

        full_init = [g0, vbeta0, logpsi0, logphi0, loggamma0, logtau0]
        full_init_shape = [item.size for item in full_init]

        free_vars, fixed_vars = ([], [])
        for item, boolean in zip(full_init, is_fixed_vars):
            if boolean:
                fixed_vars.append(item)
            else:
                free_vars.append(item)
        free_vars_shape = [item.size for item in free_vars]
        return np.concatenate(free_vars), free_vars_shape, fixed_vars

    def log_likelihood(self, y, g, beta,
                       logphi, loggamma, logtau,
                       eval_gradient=False, **kwargs):
        """Log-likelihood of the data

        Parameters
        ----------

        y : array

        g : array

        beta : array

        logphi : array

        loggamma : array

        logtau : array

        eval_gradient : boolean

        Returns
        -------

        res : float,

        Notes
        -----
        Returns the log-likelihood

        .. math::

            \ln p(\mathbf{y} \mid \mathbf{g},
            \\boldsymbol{\\beta},
            \ln \\boldsymbol{\phi},
            \ln \\boldsymbol{\gamma},
            \ln \\boldsymbol{\\tau})

        of the vectorised data :math:`\mathbf{y}`. Although note the
        for the GP hyperparameters, temperature parameter
        :math:`\boldsymbol{\gamma}` and data precision
        :math:`boldsymbol{\tau}` that function takes as arguments the
        log transform of these values (and consequently returns the
        gradient wth respect to the log transformed parameters) to
        ensure a natural positivity constraints.
        """
        if eval_gradient:
            # ODE model inv. covariance matrix            
            Lam, Lam_g_grad, Lam_beta_grad, Lam_logphi_grad, Lam_gam_grad = \
                 Lambda(g, beta, logphi, np.exp(loggamma), self, True)

        else:
            Lam = Lambda(g, beta, logphi, np.exp(loggamma), self, False)
        # invert the ODE model covariance matrix
        Lode = np.linalg.cholesky(Lam)
        Kode = cho_solve((Lode, True), np.eye(Lode.shape[0]))

        # data covariance
        tau = np.exp(logtau)        
        Kdat = block_diag(*[1 / tauk * np.eye(self._Ndata)
                            for tauk in tau])

        if self.is_tt_aug:
            # got to select the subindices of Kode
            _data_inds = np.concatenate([self.data_inds + self.dim.N*k
                                         for k in range(self.dim.K)])
            _data_slice = np.ix_(_data_inds, _data_inds)
            K = Kode[_data_slice] + Kdat
        else:
            K = Kode + Kdat
        Kchol = np.linalg.cholesky(K)

        # evaluate the loglikelihood

        # Support multi-dimensional output of self.y_train_ (as in sklean GPR api)
        if len(y.shape) == 1:
            y_train = y[:, np.newaxis]
        else:
            y_train = y

        alpha = cho_solve((Kchol, True), y_train)

        log_lik_dims = -.5 * np.einsum("ik, ik->k",y_train, alpha)
        log_lik_dims -= np.log(np.diag(Kchol)).sum()
        log_lik_dims -= K.shape[0] / 2 * np.log(2 * np.pi)
        log_lik = log_lik_dims.sum(-1)  # sum over output dimensions

        if eval_gradient:

            K_g_grad = [-cho_solve((Lode, True),
                                   cho_solve((Lode, True), Lam_g_grad[..., i]).T)
                        for i in range(g.size)]
            K_b_grad = [-cho_solve((Lode, True),
                                   cho_solve((Lode, True), Lam_beta_grad[..., i]).T)
                        for i in range(beta.size)]
            
            K_lphi_grad = [-cho_solve((Lode, True),
                                      cho_solve((Lode, True), Lam_logphi_grad[..., i]).T)
                           for i in range(logphi.size)]
            K_gam_grad = [-cho_solve((Lode, True),
                                     cho_solve((Lode, True),
                                               Lam_gam_grad[..., i]).T)
                          for i in range(loggamma.size)]

            # if the time vector has been augmented we
            # need to map everything back to data inds
            if self.is_tt_aug:
                K_g_grad = np.dstack([arr[_data_slice]
                                      for arr in K_g_grad])
                K_b_grad = np.dstack([arr[_data_slice]
                                      for arr in K_b_grad])
                K_lphi_grad = np.dstack([arr[_data_slice]
                                         for arr in K_lphi_grad])
                K_gam_grad = np.dstack([arr[_data_slice]
                                        for arr in K_gam_grad])
            else:
                K_g_grad = np.dstack(K_g_grad)
                K_b_grad = np.dstack(K_b_grad)
                K_lphi_grad = np.dstack(K_lphi_grad)
                K_gam_grad = np.dstack(K_gam_grad)

            # gradient of K with respect to data precisions
            K_tau_grad = []
            dek = np.zeros(self.dim.K)
            for k in range(self.dim.K):
                dek[k] = -1/tau[k]**2
                K_tau_grad.append(np.kron(np.diag(dek),
                                          np.eye(self._Ndata)))
                dek[k] = 0.
            K_tau_grad = np.dstack(K_tau_grad)

            tmp = np.einsum("ik,jk->ijk", alpha, alpha)  # k: output-dimension
            tmp -= cho_solve((Kchol, True), np.eye(Kchol.shape[0]))[:, :, np.newaxis]

            log_lik_g_grad_dims = .5 * np.einsum("ijl,ijk->kl", tmp, K_g_grad)
            log_lik_g_grad = log_lik_g_grad_dims.sum(-1)

            log_lik_b_grad_dims = .5 * np.einsum("ijl,ijk->kl", tmp, K_b_grad)
            log_lik_b_grad = log_lik_b_grad_dims.sum(-1)

            log_lik_lphi_grad_dims = .5 * np.einsum("ijl,ijk->kl", tmp, K_lphi_grad)
            log_lik_lphi_grad = log_lik_lphi_grad_dims.sum(-1)
            
            log_lik_lgam_grad_dims = .5 * np.einsum("ijl,ijk->kl", tmp, K_gam_grad)
            log_lik_lgam_grad = log_lik_lgam_grad_dims.sum(-1) * np.exp(loggamma)

            log_lik_ltau_grad_dims = .5 * np.einsum("ijl,ijk->kl", tmp, K_tau_grad)
            log_lik_ltau_grad = log_lik_ltau_grad_dims.sum(-1) * np.exp(logtau)

            return log_lik, \
                   log_lik_g_grad, \
                   log_lik_b_grad, \
                   log_lik_lphi_grad, \
                   log_lik_lgam_grad, \
                   log_lik_ltau_grad

        else:
            return log_lik

    def _prior_logpdf(self, g, logpsi, eval_gradient=False, **kwargs):
        """Logpdf of the prior
        """
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

    @property
    def _Ndata(self):
        if self.is_tt_aug:
            return len(self.data_inds)
        else:
            return self.dim.N


class GibbsMLFMAdapGrad(MLFMAdapGrad):
    """
    Extends the MLFM-AG matching model to implement
    conditional Gibbs sampling.

    Parameters
    ----------

    basis_mats : tuple of square ndarray
        A tuple of square numpy array_like with the same shape.

    R : int
        Number of latent forces in the model.

    lf_kernels : list of kernel objects, optional
        Kernels of the latent force Gaussian process objects. If None
        is passed, the kernel \"1.0 * RBF(1.0)\" is used for the R
        latent forces.
    """
    def __init__(self, *args, **kwargs):
        super(GibbsMLFMAdapGrad, self).__init__(*args, **kwargs)
        
    def gibbsfit(self, times, Y,
                 sample=('g'),
                 size=100,
                 burnin=100,
                 **kwargs):

        N_outputs = Y.shape[1]

        if len(sample) == 0:
            return {}

        try:
            mapres = kwargs['mapres']
            # still need to call false setup
            self._setup(times, **kwargs)
        except:
            # get the results from the MAP fit of the model
            mapres = self.fit(times, Y, **kwargs)

        samples = {key: [] for key in sample}

        nt = 0  # iteration counter

        # initalise values at the MAP estimates
        bcur = mapres.beta
        gcur = mapres.g.ravel()

        # check for a prior on beta
        try:
            beta_prior = kwargs['beta_prior']
            
        except KeyError:
            raise ValueError("Gibbs sampling requires specifying a "
                             "prior on beta")
            
        
        while len(samples[sample[0]]) < size:
            Ex, Covx = self.x_condpdf(bcur, gcur,
                                      logphi=mapres.logphi,
                                      loggamma=mapres.loggamma,
                                      include_data=True,
                                      logtau=mapres.logtau,
                                      vecy=Y,
                                      same_ic=False)
            xcur = np.column_stack((
                multivariate_normal.rvs(ex, Covx) for ex in Ex.T))

            if 'x' in sample:
                if nt > burnin:
                    samples['x'].append(xcur)            

            if 'g' in sample:
                Eg, Covg = self.g_condpdf_mo(xcur, bcur,
                                             gamma=np.exp(mapres.loggamma),
                                             logphi=mapres.logphi,
                                             logpsi=mapres.logpsi)
                gcur = multivariate_normal.rvs(Eg, Covg)
                if nt > burnin:
                    samples['g'].append(gcur)

            if 'beta' in sample:
                Eb, Covb = self.beta_condpdf_mo(xcur, gcur,
                                                gamma=np.exp(mapres.loggamma),
                                                logphi=mapres.logphi)
                bcur = multivariate_normal.rvs(mean=Eb, cov=Covb)
                bcur = bcur.reshape((self.dim.D, self.dim.R+1)).T
                if nt > burnin:
                    samples['beta'].append(bcur.ravel())

            nt += 1

        # convert sample list to arrays
        for key, item in samples.items():
            samples[key] = np.asarray(item)

        return samples

    def g_condpdf_mo(self, vecx, beta, gamma=None, logphi=None, logpsi=None, **kwargs):
        ## handle parameter construct from arguments
        if logphi is None and gamma is not None \
           or logphi is not None and gamma is None:
            raise ValueError("gamma and logphi must either be set together or both None.")

        elif logphi is None and gamma is None:
            # if logphi hasn't been passed then
            try:
                # the model matrices depending on logphi
                phi_depmats = kwargs.pop('phi_depmats')
                Lxx_list, Mdx_list, Schol_list = \
                          phi_depmat
            except:
                raise ValueError

        else:
            logphi_shape = _get_gp_theta_shape(self.latentstates)
            logphi = _unpack_vector(logphi, logphi_shape)

        A = np.asarray([sum(brd*Lrd for brd, Lrd in zip(br, self.basis_mats))
                        for br in beta])

        invcov, premean = ([], [])

        # sum over each dimension
        for k in range(self.dim.K):

            # Get the covariance matrices which do not depend on
            # the values of x_k^{(m)}
            if logphi is None:
                Lxx, Mdx, Schol = Lxx_list[k], \
                                  Mdx_list[k], \
                                  Schol_list[k]
            else:
                _phidep_covs = _ls_covar_k_wgrad(logphi[k],
                                                 gamma[k],
                                                 self.ttc,
                                                 self.latentstates[k])
                Lxx, Mdx, Schol = _phidep_covs[0]

            # sum over all of the outputs
            for m in range(vecx.shape[1]):

                Xm = vecx[:, m].reshape(self.dim.K, self.dim.N)
                vk = vk_flow_rep(k,
                                 Xm,
                                 A)

                mk = Mdx.dot(Xm[k, :])
                SkinvVk = np.column_stack(
                    [cho_solve((Schol, True), np.diag(vkr))
                     for vkr in vk[1:]])

                Vk = np.row_stack([np.diag(vkr) for vkr in vk[1:]])
                ic = Vk.dot(SkinvVk)

                premean.append(SkinvVk.T.dot(mk-vk[0]))
                invcov.append(ic)

        # Add the contribution from the prior
        prior_ic = []

        if logpsi is None:
            try:
                # The model matrices depending on logpsi
                prior_ic = kwargs.pop('gprior_invcovs')
            except:
                raise ValueError("If 'logpsi' is None then the"
                                 " prior covariance matrix of g"
                                 " must be provided")
        else:
            logpsi_shape = _get_gp_theta_shape(self.latentforces)
            logpsi = _unpack_vector(logpsi, logpsi_shape)
            for theta, gp in zip(logpsi, self.latentforces):
                kern = gp.kernel.clone_with_theta(theta)
                Cgr = kern(self.ttc[:, None])
                Cgr[np.diag_indices_from(Cgr)] += gp.alpha
                L = np.linalg.cholesky(Cgr)

                prior_ic.append(cho_solve((L, True), np.eye(L.shape[0])))

        invcov = sum(invcov) + block_diag(*prior_ic)
        cov = np.linalg.inv(invcov)
        mean = cov.dot(sum(premean))

        return mean, cov

    def g_condpdf(self, vecx, beta, gamma=None, logphi=None, logpsi=None, **kwargs):
        """
        Conditional distribution of the latent force.

        Parameters
        ----------

        vecx : numpy array, shape (N*K, )
            Vectorised state variables.

        beta : numpy array, shape (R+1, D)
        """

        ## handle parameter construct from arguments
        if logphi is None and gamma is not None \
           or logphi is not None and gamma is None:
            raise ValueError("gamma and logphi must either be set together or both None.")

        elif logphi is None and gamma is None:
            # if logphi hasn't been passed then
            try:
                # the model matrices depending on logphi
                phi_depmats = kwargs.pop('phi_depmats')
                Lxx_list, Mdx_list, Schol_list = \
                          phi_depmat
            except:
                raise ValueError

        else:
            logphi_shape = _get_gp_theta_shape(self.latentstates)
            logphi = _unpack_vector(logphi, logphi_shape)

        A = np.asarray([sum(brd*Lrd for brd, Lrd in zip(br, self.basis_mats))
                        for br in beta])

        # handy shapes
        Xrowform = vecx.reshape(self.dim.K, self.dim.N)
        invcov, premean = ([], [])
        for k in range(self.dim.K):

            vk = vk_flow_rep(k, Xrowform, A)

            if logphi is None:
                # get the phi dependent matrices from earlier
                Lxx, Mdx, Schol = Lxx_list[k], \
                                  Mdx_list[k], \
                                  Schol_list[k]
            else:
                _phidep_covs =  _ls_covar_k_wgrad(logphi[k],
                                                   gamma[k],
                                                   self.ttc,
                                                   self.latentstates[k])
                Lxx, Mdx, Schol = _phidep_covs[0]
            mk = Mdx.dot(Xrowform[k, :])
            SkinvVk = np.column_stack(
                [cho_solve((Schol, True), np.diag(vkr))
                 for vkr in vk[1:]])

            Vk = np.row_stack([np.diag(vkr) for vkr in vk[1:]])
            ic = Vk.dot(SkinvVk)

            premean.append(SkinvVk.T.dot(mk-vk[0]))
            invcov.append(ic)
        
        # add the contribution from the prior
        prior_ic = []

        if logpsi is None:
            try:
                # the model matries depending on logpsi
                prior_ic = kwargs.pop('gprior_invcovs')
            except:
                raise ValueError
        else:
            logpsi_shape = _get_gp_theta_shape(self.latentforces)
            logpsi = _unpack_vector(logpsi, logpsi_shape)
            for theta, gp in zip(logpsi, self.latentforces):
                kern = gp.kernel.clone_with_theta(theta)
                Cgr = kern(self.ttc[:, None])
                Cgr[np.diag_indices_from(Cgr)] += gp.alpha
                L = np.linalg.cholesky(Cgr)

                prior_ic.append(cho_solve((L, True), np.eye(L.shape[0])))

        invcov = sum(invcov) + block_diag(*prior_ic)

        cov = np.linalg.inv(invcov)
        mean = cov.dot(sum(premean))

        return mean, cov

    def beta_condpdf_mo(self, vecx, vecg, gamma=None, logphi=None, **kwargs):
        ## handle parameter construct from arguments
        if logphi is None and gamma is not None \
           or logphi is not None and gamma is None:
            raise ValueError("gamma and logphi must either be set together or both None.")

        elif logphi is None and gamma is None:
            # if logphi hasn't been passed then
            try:
                # the model matrices depending on logphi
                phi_depmats = kwargs.pop('phi_depmats')
                Lxx_list, Mdx_list, Schol_list = \
                          phi_depmat
            except:
                raise ValueError

        else:
            logphi_shape = _get_gp_theta_shape(self.latentstates)
            logphi = _unpack_vector(logphi, logphi_shape)

        invcov, premean = ([], [])

        # sum over each dimension
        for k in range(self.dim.K):
            # Get the covariance matrices which do not depend on
            # the values of x_k^{(m)}
            if logphi is None:
                Lxx, Mdx, Schol = Lxx_list[k], \
                                  Mdx_list[k], \
                                  Schol_list[k]
            else:
                _phidep_covs = _ls_covar_k_wgrad(logphi[k],
                                                 gamma[k],
                                                 self.ttc,
                                                 self.latentstates[k])
                Lxx, Mdx, Schol = _phidep_covs[0]

            Skinv = cho_solve((Schol, True), np.eye(Schol.shape[0]))

            
            # sum over all of the outputs
            for m in range(vecx.shape[1]):

                Xm = vecx[:, m].reshape(self.dim.K, self.dim.N)
                mk = Mdx.dot(Xm[k, :])
                Skinvmk = Skinv.dot(mk)

                wk = wk_flow_rep(k, Xm, vecg.reshape(self.dim.R, self.dim.N),
                                 self.basis_mats)
                Wk = np.column_stack(wk)

                invcov.append(Wk.T.dot(Skinv.dot(Wk)))
                premean.append(np.array([w.dot(Skinvmk) for w in wk]))

        invcov = sum(invcov) + \
                 np.diag(np.ones((self.dim.R+1)*self.dim.D) / 10.**2)
        cov = np.linalg.inv(invcov)
        mean = cov.dot(sum(premean))
        return mean, cov

    def beta_condpdf(self, vecx, vecg, gamma=None, logphi=None, **kwargs):

        invcov, premean = ([], [])
        X = vecx.reshape(self.dim.K, self.dim.N)

        if logphi is not None:
            logphi = _unpack_vector(logphi, _get_gp_theta_shape(self.latentstates))

        for k in range(self.dim.K):

            if logphi is None:
                ## get the phi dependent matrices from earlier
                Lxx, Mdx, Schol = Lxx_list[k], \
                                  Mdx_list[k], \
                                  Schol_list[k]
            else:
                _phidep_covs = _ls_covar_k_wgrad(logphi[k],
                                                 gamma[k],
                                                 self.ttc,
                                                 self.latentstates[k])
                Lxx, Mdx, Schol = _phidep_covs[0]
            
            mk = Mdx.dot(X[k, :])

            wk = wk_flow_rep(k,
                             X,
                             vecg.reshape(self.dim.R, self.dim.N),
                             self.basis_mats)
            Wk = np.column_stack(wk)


            Skinv = cho_solve((Schol, True), np.eye(self.dim.N))

            Skinvmk = Skinv.dot(mk)
            pm = np.array([w.dot(Skinvmk) for w in wk])

            invcov.append(Wk.T.dot(Skinv.dot(Wk)))
            premean.append(pm)

        # add the contribution from the prior
        invcov = sum(invcov) + \
                 np.diag(np.ones((self.dim.R+1)*self.dim.D) / 10.**2)
        cov = np.linalg.inv(invcov)
        mean = cov.dot(sum(premean))
        return mean, cov

    def x_condpdf(self, beta, vecg, logphi, loggamma, same_ic=True, **kwargs):
        Lam = Lambda(vecg, beta, logphi, np.exp(loggamma), self, False)

        include_data = kwargs.pop("include_data", False)
        if include_data:
            try:
                logtau = kwargs.pop("logtau")
                tau = np.exp(logtau)
                data_invcov = np.kron(np.diag(tau), np.eye(self._Ndata))

                try:

                    y = kwargs.pop("vecy")

                    if same_ic:
                        premean = data_invcov.dot(y.sum(-1))
                        invcov = Lam + y.shape[1]*data_invcov
                    else:
                        premean = data_invcov.dot(y)
                        invcov = Lam + data_invcov

                    cov = np.linalg.inv(invcov)
                    mean = cov.dot(premean)
                    return mean, cov
                    
                except KeyError:
                    raise ValueError("Must provide (vectorised) data: vecy")
                    
            except KeyError:
                raise ValueError("Must provide log precisions: logtau")
        
        return np.zeros(self.dim.N*self.dim.K), np.linalg.inv(Lam)

    def x_condpdf_mo(self, beta, vecg, logphi, loggamma, **kwargs):
        """
        Possibility of multiple outputs
        """
        Lam = Lambda(vecg, beta, logphi, np.exp(loggamma), self, False)

        include_data = kwargs.pop("include_data", False)
        if include_data:

            logtau = kwargs.pop("logtau")
            tau = np.exp(logtau)

            data_invcov = np.zeros((self.dim.N*self.dim.K,
                                    self.dim.N*self.dim.K))
            premean = np.zeros(self.dim.N*self.dim.K)

            Ys = kwargs.pop('Ys')
            data_inds = kwargs.pop('data_inds')
            n_outputs = len(Ys)  # Ys should be list lik
            
            for q, vecy in enumerate(Ys):
                dind = data_inds[q]
                dic_q = np.kron(np.diag(tau), np.eye(len(dind)))
                
                # pad indices with extra dimension
                dind = np.asarray(dind)
                dind = np.concatenate([dind + self.dim.N*k
                                       for k in range(self.dim.K)])
                data_invcov[np.ix_(dind, dind)] += dic_q

                premean[dind] += dic_q.dot(vecy)

            invcov = Lam + data_invcov
            cov = np.linalg.inv(invcov)
            mean = cov.dot(premean)

            return mean, cov


class VarMLFMAdapGrad(MLFMAdapGrad):
    def __init__(self, *args, **kwargs):
        super(VarMLFMAdapGrad, self).__init__(*args, **kwargs)
        self.basis_mats = np.array(self.basis_mats)

    def predict(self, tt, return_std=False):
        pass

    def varfit(self, times, Y, gtol=1e-3, max_iter=100, **kwargs):

        # get the results from fitting the MAP model
        mapres = self.fit(times, Y, **kwargs)

        # initalise the distribution of g at the MAP
        Eg = mapres.g.ravel()
        Covg = np.diag([0.1]*Eg.size)

        Eb = mapres.beta.ravel()
        if kwargs.pop('beta_is_fixed', False):
            Covb = np.zeros((Eb.size, Eb.size))
        else:
            Covb = np.diag([0.1]*Eb.size)

        Deltg = np.inf  # difference between prev and cur g

        nt = 0 # Count number of iterations
        while True:
            Ex, Covx = self.x_cond(Eg, Covg, Eb, Covb,
                                   mapres.logphi, np.exp(mapres.loggamma),
                                   include_data=True, Y=Y, tau=np.exp(mapres.logtau))
            Eg_, Covg_ = self.g_cond(Ex, Covx, Eb, Covb,
                                     mapres.logphi, np.exp(mapres.loggamma),
                                     mapres.logpsi)
            # Calculate distance moved
            Deltg = np.linalg.norm(Eg - Eg_)
            # Update 
            Eg = Eg_
            Covg = Covg_
            # Check converg. conditions
            if Deltg < gtol:
                break
            nt += 1
            if nt >= max_iter:
                break

        # store results
        self.Ex_train_ = Ex.copy()
        self.Lx_train_ = np.linalg.cholesky(Covx)
        
        # return Ex in a more resonable shape
        #Ex = Ex.reshape(self.dim.K, self.dim.N, Y.shape[1]).transpose(1, 0, 2)

        #self.g_L_chol_ = cholesky(Covg, lower=True)
        #self.g_alpha_ = cho_solve((self.g_L_chol_, True), Eg)

        # Sensible reshaping
        Eg = Eg.reshape(self.dim.R, self.dim.N).T
        Covg = Covg.reshape(self.dim.N, self.dim.N, self.dim.R, self.dim.R)
        Covg = Covg.transpose(1, 0, 3, 2)
        
        return mapres, Eg, Covg

        
    def g_cond(self, Ex, Covx, Eb, Covb,
               logphi, gamma, logpsi, **kwargs):

        # reshape logphi
        logphi = _unpack_vector(logphi, _get_gp_theta_shape(self.latentstates))

        premean = 0.
        invcov = np.zeros((self.dim.R*self.dim.N, self.dim.R*self.dim.N))

        if len(Ex.shape) == 1:
            Ex = Ex[:, None]

        ExxTs = [Covx + np.outer(Ex_q, Ex_q) for Ex_q in Ex.T]
        EbbT = Covb + np.outer(Eb, Eb)

        # EA
        Eb = Eb.reshape((self.dim.R+1, self.dim.D))
        EA = np.array([sum(Ebrd*Ld for Ebrd, Ld in zip(Ebr, self.basis_mats))
                       for Ebr in Eb])

        # Useful variables
        N = self.dim.N

        for k in range(self.dim.K):

            LxxMdxSchol = _ls_covar_k_wgrad(logphi[k],
                                            gamma[k],
                                            self.ttc,
                                            self.latentstates[k])
            Lxx, Mdx, Schol = LxxMdxSchol[0]
            Skinv = cho_solve((Schol, True), np.eye(self.dim.N))

            # sum over possibly multimensional output of state variable
            for ExxT in ExxTs:
                # E[vkr xkT]
                Evkr_mkT, Evkr_vk0 = ([], [])

                for r in range(1, self.dim.R+1):
                    Evkr_mkT.append(sum(EA[r, k, j]*ExxT[j*N:(j+1)*N,
                                                         k*N:(k+1)*N].dot(Mdx.T)
                                        for j in range(self.dim.K)))

                    Evkr_vk0.append(EvkrvksT(k, r, 0, ExxT, EbbT, self.basis_mats))
                
                    for s in range(1, self.dim.R+1):
                        expr = EvkrvksT(k, r, s, ExxT, EbbT, self.basis_mats)

                        invcov[(r-1)*self.dim.N:r*self.dim.N,
                               (s-1)*self.dim.N:s*self.dim.N] += expr * Skinv

                pmk = [np.sum( (_Evkr_mk - _Evkr_v0) * Skinv, axis=1)
                       for _Evkr_mk, _Evkr_v0 in zip(Evkr_mkT, Evkr_vk0)]
                pmk = np.concatenate(pmk)

                premean += pmk

        # Add contribution from prior
        logpsi = _unpack_vector(logpsi, _get_gp_theta_shape(self.latentforces))
        for r in range(self.dim.R):
            gp = self.latentforces[r]
            kern = gp.kernel.clone_with_theta(logpsi[r])
            Kr = kern(self.ttc[:, None])
            Kr[np.diag_indices_from(Kr)] += 1e-5
            Lr = np.linalg.cholesky(Kr)

            invcov[r*self.dim.N:(r+1)*self.dim.N,
                   r*self.dim.N:(r+1)*self.dim.N] += cho_solve((Lr, True), np.eye(self.dim.N))
        cov = np.linalg.inv(invcov)
        mean = cov.dot(premean)

        return mean, cov

    def x_cond(self,
               Eg, Covg, Eb, Covb,
               logphi, gamma, **kwargs):

        # contribution from model
        invcov = E_qGqB_Lambdag(Eg, Covg, Eb, Covb, self,
                                logphi, gamma, **kwargs)

        # potential contribution from data
        include_data = kwargs.pop('include_data', False)
        if include_data:

            try:
                tau = kwargs.pop('tau')
            except KeyError:
                msg = "If including the data contribution must pass precisions tau"
                raise ValueError(msg)

            datainv_cov = np.kron(np.diag(tau), np.eye(self._Ndata))

            try:
                Y = kwargs.pop('Y')  # assumed to be already shaped
                if len(Y.shape) == 1:
                    Y = Y[:, None]                    
            except KeyError:
                raise ValueError("If including data must include data 'Y'")

            if self.is_tt_aug:
                _data_inds = np.array(self.data_inds)
                _data_inds = np.concatenate([_data_inds + self.dim.N*k
                                             for k in range(self.dim.K)])

                pY = np.zeros((self.dim.N*self.dim.K, Y.shape[1]))
                pY[_data_inds, :] = Y

                ic = np.zeros(invcov.shape)
                ic[np.ix_(_data_inds, _data_inds)] = datainv_cov
                
                invcov += ic
                premean = ic.dot(pY)

            else:
                invcov += datainv_cov
                premean = datainv_cov.dot(Y)
        cov = np.linalg.inv(invcov)

        if include_data:
            mean = cov.dot(premean)
        else:
            mean = np.zeros(self.dim.N*self.dim.K)

        return mean, cov

    def beta_cond(self, Ex, Covx, Eg, Covg,
                  logphi, gamma, **kwargs):

        # reshape logphi
        logphi = _unpack_vector(logphi, _get_gp_theta_shape(self.latentstates))

        if len(Ex.shape) == 1:
            Ex = Ex[:, None]
        ExxTs = [Covx + np.outer(Ex_q, Ex_q) for Ex_q in Ex.T]

        Eg = np.concatenate((np.ones(self.dim.N), Eg))
        Covg = block_diag(np.zeros((self.dim.N, self.dim.N)),
                          Covg)
        EggT = Covg + np.outer(Eg, Eg)

        N, K, R, D = self.dim

        premean = np.zeros(D*(R+1))
        invcov = np.zeros((D*(R+1), D*(R+1)))

        for k in range(self.dim.K):

            LxxMdxSchol = _ls_covar_k_wgrad(logphi[k], gamma[k],
                                            self.ttc, self.latentstates[k])
            Lxx, Mdx, Schol = LxxMdxSchol[0]
            Skinv = cho_solve((Schol, True), np.eye(self.dim.N))
            SkinvMk = Skinv.dot(Mdx)

            for ExxT in ExxTs:

                for r1 in range(R+1):
                    for d1 in range(D):
                        for r2 in range(R+1):
                            for d2 in range(D):
                                wmwk = EwkrwksT(k, r1, r2, d1, d2, ExxT, EggT, self.basis_mats)
                                # Tr{ wwT Skinv }
                                invcov[r1*D + d1,
                                       r2*D + d2] += np.einsum('ij,ji->', wmwk.T, Skinv)

                        sum_ExxT = sum(self.basis_mats[d1, k, j]*ExxT[k*N:(k+1)*N, j*N:(j+1)*N]
                                       for j in range(K))
                        sum_ExxT = sum_ExxT * Eg[r1*N:(r1+1)*N][None, :]
                        premean[r1*D + d1] += np.einsum('ij,ji->', sum_ExxT, SkinvMk)

        invcov[np.diag_indices_from(invcov)] += 1/10**2

        L = np.linalg.cholesky(invcov)

        Covb = cho_solve((L, True), np.eye(L.shape[0]))
        Eb = Covb.dot(premean)

        return Eb, Covb
        

"""
Flow Function Representations
-----------------------------

Representations of the linear flow function.
"""

def uk_flow_rep(k, g, A):
    """
    Representation as [A(ti)x]_k = sum_j=1^k u_kj o x_j

    Parameters
    ----------

    k : int
        Component of the flow
    
    g : array_like, shape (R, N)
        Rows of the latent forces g_1,...,g_R

    A : list
        list of square K x K structure matrices
    """
    return [A[0, k, j] + sum(ar[k, j]*gr for ar, gr in zip(A[1:], g))
            for j in range(A[0].shape[0])]

def vk_flow_rep(k, x, A):
    """
    Representation as [A(ti)x]_k = sum_{r=0}^R v_{kr} o g_r

    Parameters
    ----------

    k : int

    x : array_like, shape (K, N)

    A : list
        list of square K x K structure matrices
    """
    vk0 = sum(A[0, k, j]*xj for j, xj in enumerate(x))
    vkr = [sum(ar[k, j]*xj for j, xj in enumerate(x)) for ar in A[1:]]
    return [vk0] + vkr

def EvkrvksT(k, r, s, ExxT, EbbT, L):
    """
    Returns EvkrvksT.
    """
    K = L.shape[-1]
    N = ExxT.shape[0] // K
    D = L.shape[0]

    vkrvksT = 0.
    for i in range(K):
        for j in range(K):
            lki = L[:, k, i]
            lkj = L[:, k, j]

            EArkiAskj = lki.dot(EbbT[r*D:(r+1)*D,
                                     s*D:(s+1)*D].dot(lkj))

            vkrvksT += EArkiAskj*ExxT[i*N:(i+1)*N,
                                      j*N:(j+1)*N]
    return vkrvksT

def wk_flow_rep(k, x, g, L):
    """
    Representation as [A(ti)x]_k =

    [..., w_{0d}, w_{1d},..., w_{Rd}, ...]
    """
    D = len(L)
    R = g.shape[0]
    wk = []
    for d in range(D):
        for r in range(R+1):
            if r == 0:
                gr = 1.
            else:
                gr = g[r-1, :]
            wkrd = sum(L[d][k, j] * xj for j, xj in enumerate(x)) * gr
            wk.append(wkrd)
    return wk

def EwkrwksT(k, r, s, d1, d2, ExxT, EggT, L):
    """
    Slower vesion of this function
    """
    K = L.shape[-1]
    N = ExxT.shape[0] // K
    D = L.shape[0]

    EgrgsT = EggT[r*N:(r+1)*N, s*N:(s+1)*N]

    wkrwksT = 0.
    for i in range(K):
        for j in range(K):
            wkrwksT += ExxT[i*N:(i+1)*N, j*N:(j+1)*N]*L[d1, k, i]*L[d2, k, j]

    return EgrgsT * wkrwksT

"""
ODE Probability Model
---------------------

ODE parameter dependent parts of the ODE function.
"""

def Lambdag_k(k, vecg, beta, logphi_k, gamma_k, mlfm,
              eval_gradient=True):
    """
    kth force dependent contribution to the model covariance function.
    """
    # structure matrices
    if beta is None:
        A = np.asarray(
            [np.zeros((mlfm.dim.K, mlfm.dim.K)),
            *mlfm.basis_mats])
    else:
        A = np.asarray([sum(brd*Lrd for brd, Lrd in zip(br, mlfm.basis_mats))
                        for br in beta])

    covs, grads = _ls_covar_k_wgrad(logphi_k, gamma_k,
                                    mlfm.ttc, mlfm.latentstates[k],
                                    return_Cxx_inv_grad=True)
    # unpack cov and grads
    Lxx, Mdx, Schol = covs
    Cxx_inv_grad, Mdx_grad, S_grad = grads

    # linear repr. of the flow as a function of g
    uk = uk_flow_rep(k, vecg.reshape(mlfm.dim.R, mlfm.dim.N), A)

    diagUk = [np.diag(uki) for uki in uk]
    diagUk[k] -= Mdx
    Skinv_Uk = np.column_stack([cho_solve((Schol, True), Dkj) for Dkj in diagUk])

    lamk = np.row_stack([Dki.T for Dki in diagUk]).dot(Skinv_Uk)

    # add the contribution from the prior, (Cxx inverse)
    lamk[k*mlfm.dim.N:(k+1)*mlfm.dim.N,
         k*mlfm.dim.N:(k+1)*mlfm.dim.N] += cho_solve((Lxx, True),
                                                     np.eye(Lxx.shape[0]))

    if eval_gradient:

        # gradient wrt to g
        lamk_g_gradient = []
        en = np.zeros(mlfm.dim.N)
        for r in range(mlfm.dim.R):
            for n in range(mlfm.dim.N):
                en[n] = 1.  # std. basis vector
                # gradient of diag(uk) wrt g_{rn}
                dUk_grn = [np.diag(A[r+1, k, j]*en) for j in range(mlfm.dim.K)]
                expr = np.row_stack(dUk_grn).dot(Skinv_Uk)
                lamk_g_gradient.append((expr + expr.T)[..., np.newaxis])
                # reset en
                en[n] = 0.
        lamk_g_gradient = np.dstack(lamk_g_gradient)

        # gradient wrt to logphi

        if isinstance(logphi_k, float):
            P = 1
        else:
            P = len(logphi_k)

        # gradient of Uk wrt logphi_k
        Uk_grad = np.zeros((mlfm.dim.N, mlfm.dim.N*mlfm.dim.K, P))
        for p in range(P):
            Uk_grad[:, k*mlfm.dim.N:(k+1)*mlfm.dim.N, p] -= Mdx_grad[..., p]
        expr1 = -np.stack([Skinv_Uk.T.dot(S_grad[..., p].dot(Skinv_Uk))
                           for p in range(P)], axis=2)
        expr2 = np.stack([Uk_grad[..., p].T.dot(Skinv_Uk)
                          for p in range(P)], axis=2)
        expr2t = np.stack([expr2[..., p].T
                           for p in range(P)], axis=2)
        lamk_logphik_gradient = expr1 + expr2 + expr2t
        # add the gradient wrt to prior
        for p in range(Cxx_inv_grad.shape[-1]):
            lamk_logphik_gradient[k*mlfm.dim.N:(k+1)*mlfm.dim.N,
                                  k*mlfm.dim.N:(k+1)*mlfm.dim.N,
                                  p] += Cxx_inv_grad[..., p]

        # gradient wrt to gamma_k
        lamk_gammak_gradient = -Skinv_Uk.T.dot(Skinv_Uk)[..., np.newaxis]

        # gradient wrt to beta
        if beta is not None:
            lamk_beta_gradient = []
            L = mlfm.basis_mats
            for r in range(mlfm.dim.R+1):
                if r == 0:
                    gr = np.ones(mlfm.dim.N)
                else:
                    gr = vecg[(r-1)*mlfm.dim.N:r*mlfm.dim.N]
                for d in range(mlfm.dim.D):
                    dUk_brd = [np.diag(L[d][k, j]*gr)
                               for j in range(mlfm.dim.K)]
                    expr = np.row_stack(dUk_brd).dot(Skinv_Uk)
                    lamk_beta_gradient.append(expr + expr.T)
            lamk_beta_gradient = np.dstack(lamk_beta_gradient)
        else:
            lamk_beta_gradient = 0. # return something summable

        return lamk, lamk_g_gradient, \
               lamk_beta_gradient, \
               lamk_logphik_gradient, lamk_gammak_gradient

    else:
        return lamk

def Lambda(vecg, beta, logphi, gamma, mlfm, eval_gradient=False):
    """
    Force dependent contribution to the model covariance function.
    """    
    logphi_shape = _get_gp_theta_shape(mlfm.latentstates)
    logphi = _unpack_vector(logphi, logphi_shape)
    if eval_gradient:
        res = ([], [], [], [], [])
        for k in range(mlfm.dim.K):
            res_k = Lambdag_k(k,
                              vecg,
                              beta,
                              logphi[k],
                              gamma[k],
                              mlfm,
                              eval_gradient=True)
            for ls, item in zip(res, res_k):
                ls.append(item)

        return sum(res[0]), \
               sum(res[1]), \
               sum(res[2]), \
               np.dstack(res[3]), \
               np.dstack(res[4])

    else:
        return sum(Lambdag_k(k,
                             vecg,
                             beta,
                             logphi[k], gamma[k],
                             mlfm, eval_gradient=False)
                   for k in range(mlfm.dim.K))

"""
Variational Inference fitting functions
=======================================
"""
def EukiukjT(k, i, j, EggT, EbbT, L):

    D = len(L)
    R = EbbT.shape[0] // D - 1
    N = EggT.shape[0] // (R + 1)

    ukiukjT = 0.
    for r in range(R+1):
        for s in range(R+1):
            EbrbsT = EbbT[r*D:(r+1)*D, s*D:(s+1)*D]
            EArkiAskj = L[:, k, i].dot(EbrbsT.dot(L[:, k, j]))
            ukiukjT += EArkiAskj * EggT[r*N:(r+1)*N, s*N:(s+1)*N]

    return ukiukjT
            
def E_qGqB_Lambdag_k(k, Eg, Covg, Eb, Covb,
                     Skinv, Mk, mlfm):
    """
    Expectation of Lambda_k w.r.t q(G)q(B)
    """
    # Useful names
    L = mlfm.basis_mats
    N, K, R, D = mlfm.dim

    # pad Eg
    _Eg = np.concatenate((np.ones(N), Eg))

    _Covg = block_diag(np.zeros((N, N)), Covg)
                        
    EggT = _Covg + np.outer(_Eg, _Eg)
    EbbT = Covb + np.outer(Eb, Eb)
    
    # reshape E[beta]
    Eb = Eb.reshape((R+1, D))
    # reshape E[g]
    Eg = Eg.reshape((R, N))    
    # get the expectation of the vectors Uki
    EA = np.array([sum(Ebrd*Ld for Ebrd, Ld in zip(Ebr, L))
                   for Ebr in Eb])

    Euk = uk_flow_rep(k, Eg, EA)

    res = np.zeros((N*K, N*K))  # to hold the result
    SkinvMk = Skinv.dot(Mk)
    
    for i in range(K):

        Eduki_SkinvkMk = np.diag(Euk[i]).dot(SkinvMk)
        
        for j in range(i+1):
            # calculate E[uki ukjT]
            E_uik_ukj_T = EukiukjT(k, i, j, EggT, EbbT, L)

            res[i*N:(i+1)*N, j*N:(j+1)*N] += E_uik_ukj_T * Skinv

            if i == k:
                res[i*N:(i+1)*N, j*N:(j+1)*N] -= \
                                 Mk.T.dot(Skinv.dot(np.diag(Euk[j])))

            if j == k:
                res[i*N:(i+1)*N, j*N:(j+1)*N] -= \
                                 np.diag(Euk[i]).dot(Skinv.dot(Mk))
            
            if i == k and j == k:
                res[i*N:(i+1)*N, j*N:(j+1)*N] += Mk.T.dot(SkinvMk)

            # Symmetric matrix
            res[j*N:(j+1)*N, i*N:(i+1)*N] = res[i*N:(i+1)*N, j*N:(j+1)*N].T

    return res

def E_qGqB_Lambdag(Eg, Covg, Eb, Covb,
                   mlfm,
                   logphi=None, gamma=None, **kwargs):

    # handle prior contribution to covariance matrices
    if logphi is not None:
        logphi = _unpack_vector(logphi, _get_gp_theta_shape(mlfm.latentstates))
    else:
        raise ValueError("Must provde logphi")

    res = 0.
    for k in range(mlfm.dim.K):

        # get the covariance matrices depending on the GP
        # interpolator of the kth latent state
        _phidep_covs = _ls_covar_k_wgrad(logphi[k],
                                         gamma[k],
                                         mlfm.ttc,
                                         mlfm.latentstates[k])
        Lxx, Mdx, Schol = _phidep_covs[0]
        Skinv = cho_solve((Schol, True), np.eye(Schol.shape[0]))

        # Calculate E[ Lambda_k(G, B) | q(G)q(B) ]
        res += E_qGqB_Lambdag_k(k, Eg, Covg, Eb, Covb,
                                Skinv, Mdx, mlfm)

        # Add contribution from the prior on latent states
        res[k*mlfm.dim.N:(k+1)*mlfm.dim.N,
            k*mlfm.dim.N:(k+1)*mlfm.dim.N] += cho_solve((Lxx, True), np.eye(mlfm.dim.N))
    return res


"""
Model Fitting Utility Functions
===============================
"""
var_names = ['g', 'beta', 'logpsi', 'logphi', 'loggamma', 'logtau']

def _fit_kwarg_parser(mlfm, **kwargs):
    # default behaviour

    # check **kwargs to see if any variables have been kept fixed
    is_fixed_vars = [kwargs.pop("".join((vn, "_is_fixed")), False)
                     for vn in var_names[:-2]]

    # Different default behaviour for logtau
    is_fixed_vars.append(kwargs.pop('loggamma_is_fixed', True))
    is_fixed_vars.append(kwargs.pop('logtau_is_fixed', True))

    return is_fixed_vars
    

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


