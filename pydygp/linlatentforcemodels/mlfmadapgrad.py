"""
Class definition for the mlfm with adaptive gradient matching
"""
import numpy as np
from .mlfm import Dimensions, BaseMLFM
from scipy.linalg import block_diag, cho_solve
from pydygp.gradientkernels import ConstantKernel, RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.optimize import minimize
from collections import namedtuple

MLFMAdapGradFitResults = namedtuple('MLFMAdapGradFitResults',
                                    'g, beta, logpsi, logphi, loggamma, tau, \
                                    optimres')

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
                 is_beta_fixed=True,
                 **kwargs):

        super(MLFMAdapGrad, self).__init__(basis_mats, **kwargs)

        # flags for fitting function
        self.is_tt_aug = False     # Has the time vector been augmented
        self.is_beta_fixed = is_beta_fixed

        # default beta behaviour
        if self.is_beta_fixed:
            assert(self.dim.R == self.dim.D)
            self.beta = np.row_stack((np.zeros(self.dim.D),
                                      np.eye(self.dim.R)))

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

    def _setup_latentstates(self, kernels=None):
        """ Handles setting up of the latent state GPs.

        Parameters
        ----------

        kernels : list, optional
            list of gradientkernel objects
        """
        if kernels is None:
            # Default is for kernels = 1.*exp(-.5*(s-t)**2)
            ls_kernels = [ConstantKernel(1.)*RBF(1.)
                          for k in range(self.dim.K)]
            self.latentstates = [GaussianProcessRegressor(kern) for kern in ls_kernels]
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
        
    def foo(self, y, vecg, beta, logphi, loggamma, tau, eval_gradient=False):
        return self.log_likelihood(y, vecg, beta, logphi,
                                   loggamma, tau,
                                   eval_gradient=eval_gradient)

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

        # for now default behaviour is fix the data precisions
        kwargs['tau_is_fixed'] = True
        kwargs['beta_is_fixed'] = self.is_beta_fixed

        # Initial preprocessing of fit args. and shape
        # to allow for parameters to be kept fixed during
        # the optimisation process
        var_names = ['g', 'beta', 'logpsi', 'logphi', 'loggamma', 'tau']

        # check **kwargs to see if any variables have been kept fixed
        is_fixed_vars = [kwargs.pop("".join((vn, "_is_fixed")), False)
                         for vn in var_names]

        # utility function to mix the free vars and the fixed vars
        def _var_mixer(free_vars, free_vars_shape, fixed_vars):
            """ Returns the ordered full variable list
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

        # vectorise y
        vecy = Y.T.ravel()

        # obj function is given by neg. log likelihood + log prior
        def objfunc(arg, free_vars_shape, fixed_vars):
            g, vbeta, logpsi, logphi, loggamma, tau = \
               _var_mixer(arg, free_vars_shape, fixed_vars)
            # reshape beta
            beta = vbeta.reshape(self.dim.R+1, self.dim.D)

            try:
                ll, ll_g_grad, ll_beta_grad, ll_lphi_grad, ll_lgam_grad = \
                    self.log_likelihood(vecy, g, beta, logphi, loggamma, tau, eval_gradient=True)
                lp, lp_g_grad, lp_lpsi_grad = \
                    self.prior_logpdf(g, logpsi, eval_gradient=True)

                grad = [-(ll_g_grad + lp_g_grad),
                        -ll_beta_grad,
                        -lp_lpsi_grad,
                        -ll_lphi_grad,
                        -ll_lgam_grad]

                grad = np.concatenate([item for item, b in zip(grad, is_fixed_vars)
                                       if not b])
                
                return -(ll + lp), grad

            except:
                return np.inf, np.zeros(arg.size)

        init, free_vars_shape, fixed_vars = \
              _fit_init(self, is_fixed_vars, **kwargs)
        res = minimize(objfunc, init,
                       jac=True,
                       args=(free_vars_shape, fixed_vars),
                       options=kwargs.pop('optim_options', None))

        # save a copy of the results from optim
        self._optim_res = res

        g_, vbeta_, logpsi_, logphi_, loggamma_, tau_ = \
            _var_mixer(res.x, free_vars_shape, fixed_vars)
        beta_ = vbeta_.reshape(self.dim.R+1, self.dim.D)

        return MLFMAdapGradFitResults(g_, beta_, logpsi_, logphi_, loggamma_, tau_, res)


    def log_likelihood(self, y, g, beta,
                       logphi, loggamma, tau,
                       eval_gradient=False, **kwargs):
        """Log-likelihood of the data

        Notes
        -----
        Returns the log-likelihood

        .. math::

            `\ln p(\mathbf{y} \mid \mathbf{g}, \boldsymbol{\phi},
            \boldsymbol{\gamma}, \boldsymbol{\tau})

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
        Kdat = block_diag(*[1 / tauk * np.eye(self.Ndata)
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
        alpha = cho_solve((Kchol, True), y)

        log_lik = -.5 * y.dot(alpha)
        log_lik -= np.log(np.diag(Kchol)).sum()
        log_lik -= K.shape[0] / 2 * np.log(2 * np.pi)

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

            tmp = np.outer(alpha, alpha)
            tmp -= cho_solve((Kchol, True), np.eye(Kchol.shape[0]))

            log_lik_g_grad = .5 * np.einsum("ij,jik", tmp, K_g_grad)
            log_lik_b_grad = .5 * np.einsum("ij,jik", tmp, K_b_grad)
            log_lik_lphi_grad = .5 * np.einsum("ij,jik", tmp, K_lphi_grad)
            log_lik_lgam_grad = .5 * np.einsum("ij,jik", tmp, K_gam_grad) * \
                                np.exp(loggamma)

            return log_lik, \
                   log_lik_g_grad, \
                   log_lik_b_grad, \
                   log_lik_lphi_grad, \
                   log_lik_lgam_grad

        else:
            return log_lik

    def prior_logpdf(self, g, logpsi, eval_gradient=False):
        """Logpdf of the prior
        """
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

        if eval_gradient:
            return ll, \
                   np.concatenate(ll_g_grad), \
                   np.concatenate(ll_logpsi_grad)

        return ll

    @property
    def Ndata(self):
        if self.is_tt_aug:
            return len(self.data_inds)
        else:
            return self.dim.N


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
Model Fitting Utility Functions
===============================
"""

def _fit_init(mlfm, is_fixed_vars, **kwargs):
    # Initalise the value of latent force GPs
    try:
        g0 = kwargs['g0']
    except KeyError:
        g0 = np.zeros(mlfm.dim.R*mlfm.dim.N)
    # Initalise the hyperparameters for the latent force GPs
    try:
        logpsi0 = kwargs['logpsi0']
    except:
        logpsi0 = np.concatenate([gp.kernel.theta
                                  for gp in mlfm.latentforces])
    # Initalise the hyperparameters for the state state GPs
    try:
        logphi0 = kwargs['logphi0']
    except:
        logphi0 = np.concatenate([gp.kernel.theta
                                  for gp in mlfm.latentstates])
    # Initalise the gradient matching error variance
    try:
        loggamma0 = kwargs['loggamma0']
    except:
        loggamma0 = np.log(1e-4*np.ones(mlfm.dim.K))
    # Initalise the data precisions
    try:
        tau0 = kwargs['tau0']
    except:
        tau0 = 1e4*np.ones(mlfm.dim.K)
    # Initalise beta
    try:
        beta0 = kwargs['beta0']
    except:
        if mlfm.is_beta_fixed:
            beta0 = mlfm.beta
        else:
            msg = "".join("If MLFMAdapGrad model initalised ",
                          "with 'beta_is_fixed = False' ",
                          "then beta0 must be passed to fit.")
            raise ValueError(msg)

    vbeta0 = beta0.ravel()  # vec. beta as [br1,...,brD]

    full_init = [g0, vbeta0, logpsi0, logphi0, loggamma0, tau0]
    full_init_shape = [item.size for item in full_init]

    free_vars, fixed_vars = ([], [])
    for item, boolean in zip(full_init, is_fixed_vars):
        if boolean:
            fixed_vars.append(item)
        else:
            free_vars.append(item)
        free_vars_shape = [item.size for item in free_vars]
    return np.concatenate(free_vars), free_vars_shape, fixed_vars

