"""
.. module:: mlfm_mh_adapgrad
"""
# External imports
from collections import namedtuple
import numpy as np
from scipy.linalg import block_diag
from scipy.stats import gamma

# Package imports
from pydygp.gaussianprocesses import GaussianProcess, GradientGaussianProcess
from pydygp.kernels import Kernel, GradientKernel

# Local imports
from . import linalg_util as la
from .mlfm import MLFM


Dimensions = namedtuple('Dimensions', 'N K R')

# Container object for the model parameter set 
MHParameters = namedtuple('MHParameters',
                          'xcur gcur')

class MLFM_MH_AdapGrad(MLFM):
    """
    This class does something related to this equation

    .. math::
       p(x) = ?
       :label: myeq

    default initalisation requires only the specification of the
    structure matrices :math:`\mathbf{A}_{r}` which make up the evolution
    equation

    .. math::
       \dot{x}(t) = \\left( A_0 + \sum_{r=1}^{R} g_r(t) A_r \\right) x(t)

    Read more in the :ref:`User guide`.

    """
    def __init__(self,
                 structure_matrices):

        # doesn't do a great deal
        super(MLFM_MH_AdapGrad, self).__init__(structure_matrices)

        self.dim = Dimensions(N=None,
                              K=self.struct_mats.shape[1],
                              R=self.struct_mats.shape[0]-1)


        """
        Initalise the container for mhparameters
        """
        self.mh_pars = MHParameters(xcur=[(None, None) for k in range(self.dim.K)],
                                    gcur=[(None, None) for r in range(self.dim.R)])

        """
        various flags for model customisation
        """
        # If true then user supplied obs probability
        # else iid diagonal Gaussians
        self.pobs_is_custom = False

        # default behaviour is a Gibbs update
        self.x_update_type = 'gibbs'

        # True if additional latent variables not corresponding
        # to the data times are to be simulated
        self.aug_latent_vars = False

        
    def time_input_setup(self, data_times, aug_times=None, nspace=1):
        """
        Handles the specification of the 'data_times' for which fitting will occur
        as well as an augmented time set 'aug_times' for prediction and complete data
        modelling

        Parameters
        ----------
        data_times : double array_like
            An array with shape (n_data, ) with the input times at which
            observations were made

        aug_times : double array_like or string, optional (default: None)
            Either an array of additional latent variables at which model
            fitting is to be carried out. Or else a string specifying a
            method to automatically generate additional latent inputs
        """
        # Attachment of data times
        self.data_times = data_times

        # Update dimensions to add number of data points
        self.dim = Dimensions(data_times.size, self.dim.K, self.dim.R)

        # handle the augmented times
        if aug_times is not None:
            add_latent_states(self, self.data_times, aug_times, nspace)

    def phi_setup(self, phi_priors=None, phi_proposals=None):
        """
        Allows the option to specify a prior and proposal distribution for the
        hyper-parameters, :math:`\mathbf{\phi}_k` for each of the :math:`K` Gaussian
        processes modelling the latent trajectories.

        Parameters
        ----------
        phi_priors : list, optional
            A list of the prior distribution objects for the collection of model
            parameters :math:`\\boldsymbol{\phi}_k` parameterising the
            :class:`pydygp.kernels.GradientKernel` objects governing the prior
            covariance of the latent trajectories. Each list element has methods:

                logpdf : callable
                    Returns the logpdf of the parameter :math:`\mathbf{\phi}_k` under
                    the prior distribution.

                rvs : callable, optional
                    Returns a random sample of :math:`\mathbf{\phi}_k` from the prior
                    distribution.

        phi_proposals : list, optional
            A list of the proposal distribution objects for the collection of model
            parameters :math:`\\boldsymbol{\psi}_k` parameterising the
            :class:`pydygp.kernels.Kernel` of the latent forces. Each list element
            has methods:

                rvs : callable
                    Returns a tuple (xnew, q_ration) where xnew is a 
        """
        self.phi_priors = phi_priors        
        self.phi_proposals = phi_proposals

    def psi_setup(self, psi_priors=None, psi_proposals=None):
        """
        Allows the option to specify a prior and proposal distribution for the
        hyper-parameters, :math:`\mathbf{\psi}_k` for each of the :math:`R` latent
        Gaussian forces
        """
        self.psi_priors = psi_priors        
        self.psi_proposals = psi_proposals

    def g_gp_setup(self,
                   kern_type='sqexp',
                   g_kernels=None):
        # initalise the kernels for each comp. of the latent trajectory
        if g_kernels is not None:
            assert(len(g_kernels) == self.dim.R)

        elif g_kernels is None:
            if kern_type=='sqexp':
                g_kernels = [Kernel.SquareExponKernel()
                             for r in range(self.dim.R)]
            else:
                raise NotImplementedError

        self.g_gps = [GaussianProcess(kern) for kern in g_kernels]
            
    def x_gp_setup(self,
                   kern_type='sqexp',
                   x_kernels=None):
                   
        """
        Handles the initalisation of the latent Gaussian Processes. This involves
        constructing the :class:`GradientKernel`

        .. note::
        
           While this initalises a list of :class:`GradientGaussianProcess` objects
           they are essentially just containers for the :class:`GradientKernel`
           because of the additional :math:`\gamma` parameters. Once the
           :class:`Kernel` is updated to include a 'nugget' variable this
           will be updated.
        
        """
        # initalise the kernels for each comp. of the latent trajectory
        if x_kernels is not None:
            assert(len(x_kernels) == self.dim.K)

        elif x_kernels is None:
            if kern_type == 'sqexp':
                x_kernels = [GradientKernel.SquareExponKernel(None, dim=1)
                             for k in range(self.dim.K)]
            else:
                raise ValueError('kern_type must be sqexp')

        # attach the x_gps
        self.x_gps = [GradientGaussianProcess(kern) for kern in x_kernels]


    """
    Allow the specification of customised distributions for
    the observation error distribution

    pobs should be of the form

        pobs(y, x, par)  | returns probability of y given x and par
    
    """
    def obs_dist_setup(self,
                       pobs=None,
                       xproposal=None,
                       pobs_par_prior=None,
                       pobs_par_proposal=None):

        if pobs is not None:
            try:
                assert(xproposal is not None)
            except:
                raise ValueError("If specifying a custom observation distribution the user must also supply custom 'xproposal' method")

            self.pobs = pobs

            # flag the new proposal distribution for x and
            # that updating is now done using a mh scheme
            self._xproposal = xproposal
            self.x_update_type = 'mh'

            # methods relating to the updating of the parameter govenrning
            # the observaion distribution
            self.pobs_par_prior = pobs_par_prior
            self.pobs_par_proposal = pobs_par_proposal

    """
    specification of the gaussian process interpolators for the
    latent trajectory
    """
    def _x_gp_setup(self,
                   kern_type='sqexp',
                   x_kernels=None,
                   kpar_proposals=None,
                   kpar_priors=None):
        """
        sets the latent gp structure for the x variables including
        the handling of 
        """

        # initalise the kernels for each comp. of the latent trajectory
        if x_kernels is not None:
            assert(len(kerns) == self.dim.K)

        elif x_kernels is None:
            if kern_type == 'sqexp':
                kerns = [GradientKernel.SquareExponKernel(None, dim=1)]
            else:
                raise ValueError('kern_type must be sqexp')

        self.x_gps = [GradientGaussianProcess(kern) for kern in x_kernels]

        """
        TODO handle the creation and assignment of the
        proposal and prior distributions
        # Proposal distribution for each of the kernel hyperparameters
        if kpar_proposals is not None:

            else:
                raise ValueError('kpar_proposals must be a list of length dim.K')

        else:
            self.x_kpar_proposal = [None]*self.dim.K

        """

    """
    Latent variable initalisation
    """
    def phi_init(self, phi_val=None, prior_rv=False):
        if prior_rv:
            # make sure the prior methods have a random variable method
            if all(p.rvs is None for p in self.phi_priors):
                raise ValueError("Prior.rvs method must be specificied for initalisation: \'prior_rv\'")

            self.phi_cur = [p.rvs() for p in self.phi_priors]

        # Add catches, but if we get to this point it should be safe to declare:
        self.is_phi_init = True

    def psi_init(self, method=None):
        if method == 'prior_rv':
            # make sure the priors have a random variable method
            if all(p.rvs is None for p in self.psi_priors):
                raise ValueError("Prior.rvs method must be specificied for initalisation: \'prior_rv\'")

            self.psi_cur = [p.rvs() for p in self.psi_priors]

        # Each latent force is multiplied by a scale parameter
        # psi_r0, the inverse of this is given the conjugate gamma prior
        self.psi_inv_scales = [1]*self.dim.R

        self.is_psi_init = True

    def g_gp_init(self, method='prior_rv', **kwargs):
        """

        Parameters
        ----------
        method : string (default is prior_rv)
           One of 'prior_rv', 'val'
        
        Initalisation of the latent forces.
        
        .. note::
           Under the default simulation strategy the latent force 'g' are
           updated first so that it is not necessary to call this method
           before fitting
        """
        if self.is_psi_init:
            if method == 'prior_rv':

                g_gp_init_from_prior(self)
            
        else:
            raise ValueError("Hyperparameters, psi, must be initalised before the G.P. values")

    def x_gp_init(self, x_val=None, prior_rv=False, method=None, **kwargs):
        # Only proceed if the gp hyperparameters have been initalised
        if self.is_phi_init:

            # Trajectory variables also require the time points
            # to have properly been initalised, including latent time points

            if method == 'data':

                # Initalises the x gp from data
                x_gp_init_from_data(self, **kwargs)
            
            # All clear, declare initalised
            self.is_x_gp_init = True
            
        else:
            raise ValueError("Hyperparameters, phi, must be initalised before the G.P. values")

    def x_gps_kpar_init(self, kpar=None):
        """
        Initalise the values of the latent GP hyperparameters
        """
        if kpars is not None:
            assert(isinstance(kpars, (list, tuple))
                   and len(kpars) == self.dim.K)
            self.x_gps_kpar_cur = [kpar for kpar in kpars] 

        else:
            # This will be updated
            raise ValueError('kpar must be a list .... ')

        # if everything has gone to plan
        self.x_gps_kpar_init = True

    """
    latent variable initalisation
    """
    def init_latent_vars(self,
                         x0s=None,
                         g0s='prior',
                         x_gps_kpar='None',
                         g_gps_kpar='None'):
        """
        Initalises the latent variables in a way
        that respects the model hierarchy
        """

        # initalise the latent trajectory gp hyperparameters
        if not self.x_gps_kpar_is_init:
            x_gps_kpar_init(x_gps_kpar)

    """
    Those functions involved in model setup
    .setup()

    handles all of the initalisation and metropolis hastings setup
    """
    def setup(self, data_times, data_Y=None, aug_times=None, **kwargs):
        # Attachment of data times
        self.data_times = data_times

        # Update dimensions to add number of data points
        self.dim = Dimensions(data_times.size, self.dim.K, self.dim.R)

        # require the 'gamma' variables initalised before
        # x may be set up
        self._gammas_cur = 0.1*np.ones(self.dim.K)

        # set up the latent variables 
        self._x_setup(aug_times, **kwargs)

    def _x_setup(self, aug_times=None, x_kern='sqexp', x_kpar=None, **kwargs):

        if aug_times is not None:
            add_latent_states(self, self.data_times, aug_times, **kwargs)

        x_gp_setup(self, x_kern, None, x_kpar)

    """
    Initalises the latent force kernel objects and their inverse covariance matrices

    Default is to initalise R SquareExponKernel objects with default parameters [1, 1]
    and to assign cur values to random draws from these objects

    the input points to which the latent forces correspond is determined
    by _x_setup and cannot be change through this method.
    """
#    def _g_setup(self, kern="sqexp", kpar=None, init_strat="prior_rv"):
#        _initalise_latent_forces(self, kern, kpar, init_strat)


    """
    sets up the inital value and simulation strategy for the 
    """
    def _pobs_par_setup(self, init_val=None, init_strategry=None, **kwargs):
        pass


    """
    set data and times
    """
    def fit(self, data_times, data_Y=None, aug_t=None):

        # attach the data time
        self.data_times = data_times

        if aug_t is not None:
            self.add_latent_states(aug_t)


        self.init_latent_vars()


    """
    X updating
    """
    def xupdate(self):
        if self.x_update_type == 'gibbs':
            self._x_gibbs_update()

        elif self.x_mh_update == 'mh':
            self._x_mh_update()


    # gibbs update of the 
    def _x_gibbs_update(self):
        _x_gibbs_update(self)
    
    """
    G updating
    """
    def g_inv_scale_update(self):
        """
        The variables :math:`\psi_{r0}` is given a conjugate gamma
        prior
        """
        a0 = 1.
        tt = self.comp_times
        b0 = 1.
        for r in range(self.dim.R):
            gr = self.g_cur[r]

            psir = self.psi_cur[r]
            psir[0] = 1.
            
            Cr = self.g_gps[r].kernel.cov(tt[:, None],
                                         kpar=psir)
            Crinv = np.linalg.inv(Cr)

            aN = a0 + tt.size
            bN = b0 + 0.5*np.dot(gr, np.dot(Crinv, gr))

            inv_scale_rv = gamma.rvs(a=aN, scale=1/bN)
            self.psi_inv_scales[r] = inv_scale_rv #aN/bN
            self.psi_cur[r][0] = 1/np.sqrt(self.psi_inv_scales[r])

    def x_gibbs_update(self):
        """
        updates the latent trajectory using gibbs sampling from the Gaussian posterior
        """
        if self.aug_latent_vars:
            tt = self.comp_times
        else:
            tt = self.data_times
            
        ms = []
        inv_covs = []

        prior_inv_covs = []
        for k in [0, 1, 2]:

            gp = self.x_gps[k]
            Mk, Sk_chol, Cxx_chol = dx_gp_condmats(tt,
                                                   gp.kernel,
                                                   self.gammas_cur[k],
                                                   kpar=self.phi_cur[k],
                                                   Cxx_return = 'chol')

            Skinv = la.back_sub(Sk_chol, np.identity(Mk.shape[0]))

            mean, inv_cov = x_cond_comp_k_par(k, self.g_cur,
                                              Skinv, Mk,
                                              self.struct_mats,
                                              self.dim.K, self.dim.R, self.dim.N)

            # Add the covariance contribution from the prior
            prior_inv_covs.append(la.back_sub(Cxx_chol, np.identity(Cxx_chol.shape[0])))

            ms.append(mean)
            inv_covs.append(inv_cov)

        # add the contribution from the prior 
        prior_inv_cov = block_diag(*prior_inv_covs)
        inv_covs.append(prior_inv_cov)
            
        # add the contribution from the data
        data_mean_contrib, data_inv_cov_contrib = data_X_conditional_contrib(self)
        ms.append(data_mean_contrib)
        inv_covs.append(data_inv_cov_contrib)

        pre_mean = sum([np.dot(ic, m) for ic, m in zip(inv_covs, ms)])

        cov = np.linalg.inv(sum(inv_covs))
        mean = np.dot(cov, pre_mean)

        cov_chol = la.cholesky(cov)

        x_new = np.dot(cov_chol, np.random.normal(size=mean.size)) + mean
        x_new = x_new.reshape(self.dim.K, self.dim.N)

        self.x_cur = [x for x in x_new]
        
        
    def g_gibbs_update(self):
        """
        updates the latent force variables using gibbs sampling from the
        Gaussian posterior
        """
        if self.aug_latent_vars:
            tt = self.comp_times
        else:
            tt = self.data_times

        ms = []
        inv_covs = []
        for k in range(self.dim.K):

            gp = self.x_gps[k]
            Mk, Sk_chol = dx_gp_condmats(tt,
                                         gp.kernel,
                                         self.gammas_cur[k],
                                         kpar=self.phi_cur[k])

            Skinv = la.back_sub(Sk_chol, np.identity(Mk.shape[0]))

            mean, inv_cov = g_cond_comp_k_par(k, self.x_cur,
                                              Skinv, Mk,
                                              self.struct_mats,
                                              self.dim.K, self.dim.R, self.dim.N)
            ms.append(mean)
            inv_covs.append(inv_cov)

        pre_mean = sum(np.dot(ic, m) for m, ic in zip(ms, inv_covs))
        sum_inv_covs = sum(inv_covs)
        cov = np.linalg.inv(sum_inv_covs)
        # mean before considering the contribution from the prior
        m1 = np.linalg.solve(sum(inv_covs), pre_mean)

        # prior covs
        prior_covs = [gp.kernel.cov(tt[:, None], kpar=psi)
                      for gp, psi in zip(self.g_gps, self.psi_cur)]

        prior_inv_cov = [np.linalg.inv(c) for c in prior_covs]
        prior_inv_cov = block_diag(*prior_inv_cov)

        inv_covs.append(prior_inv_cov)
        ms.append(np.zeros(prior_inv_cov.shape[0]))

        mean = np.linalg.solve(sum(inv_covs), pre_mean)
        cov = np.linalg.inv(sum_inv_covs)
        cov_chol = la.cholesky(cov)

        # simulate new value of g
        new_g = np.dot(cov_chol, np.random.normal(size=mean.size)) + mean

        # reshape
        new_g = new_g.reshape(self.dim.R, tt.size)

        # store new values
        self.g_cur = [g for g in new_g]


    def x_mod_cond_logpdf(self, x, g, gammas, phi):
        """
        The log of the conditional pdf of the 'model'
        """
        if self.aug_latent_vars:
            tt = self.comp_times
        else:
            tt = self.data_times

        A = self.struct_mats
        lp = 0.

        X = x.reshape(self.dim.K, self.dim.N)
        for k in [0, 1, 2]:

            uk = [A[0, k, j] + sum([arkj*gr
                                    for arkj, gr in zip(A[1:, k, j], g)])
                  for j in range(self.dim.K)]

            xk = X[k, ]
            fk = sum(ukj*xj for ukj, xj in zip(uk, X))
            gp = self.x_gps[k]
            Mk, Sk_chol, Cxx_chol = dx_gp_condmats(tt,
                                                   gp.kernel,
                                                   gammas[k],
                                                   phi[k],
                                                   Cxx_return='chol')
            etak = fk - np.dot(Mk, xk)

            expr1 = -0.5*np.dot(xk, la.back_sub(Cxx_chol, xk))
            expr2 = -0.5*np.dot(etak, la.back_sub(Sk_chol, etak))

            lp += expr1 + expr2

        return lp

"""
=======================
Model Fitting Functions
=======================
"""

def data_X_conditional_contrib(obj):
    """
    Returns the contribution to the mean and the inverse covariance of
    the distribution of X corresponding to the observed data
    """

    sigmas = obj.cur_sigmas

    if obj.aug_latent_vars:
        x_data_mean = np.zeros((obj.dim.N, obj.dim.K))
        for ind, y in zip(obj.data_inds, obj.data_Y):
            x_data_mean[ind, ] = y
        x_data_mean = x_data_mean.T.ravel()  # vectorise the mean

        sigmasq_inv_diags = []
        for s in sigmas:
            i_diag = np.zeros(obj.dim.N)
            i_diag[obj.data_inds] = 1/s**2
            sigmasq_inv_diags.append(i_diag)

        x_data_inv_cov = np.diag(
                np.concatenate(sigmasq_inv_diags)
                )
    else:
        x_data_mean = obj.data_Y.T.ravel()
        x_data_inv_cov = np.diag(
            np.concatenate([(1/s**2)*np.ones(self.dim.N) for s in sigmas])
            )

    return x_data_mean, x_data_inv_cov


def Vk_repr(Xcols, k, struct_mats):
    """
    Utility function for representing the exponential as
    
    .. math::
       v_{j}[r] = sum_{j=1}^{K} \mathbf{A}_{rkj} \mathbf{x}_j, \qquad r=1,\ldots,R
    """
    return [sum(a[k, j]*xj for j, xj in enumerate(Xcols))
            for a in struct_mats]


def g_conditional_dist(x):
    """
    ... some explanations
    """
    pass

def g_cond_comp_k_par(k, Xcols, Sinv, Mk, struct_mats, K, R, N):
    """
    
    """
    vks = Vk_repr(Xcols, k, struct_mats)
    diagV = np.row_stack([np.diag(v) for v in vks[1:]])

    inv_covar = np.row_stack((
        np.column_stack((np.outer(vi, vj)*Sinv
                         for vj in vks[1:]))
        for vi in vks[1:]))

    vec = np.dot(diagV, np.dot(Sinv, np.dot(Mk, Xcols[k]) - vks[0]))
    try:
        mean = np.linalg.solve(inv_covar, vec)
    except:
        mean = np.dot(np.linalg.pinv(inv_covar), vec)

    return mean, inv_covar

def Uk_repr(Glist, k, struct_mats):
    """
    Utility function for representing
    """
    return [sum(struct_mats[s+1, k, j]*gs + struct_mats[0, k, j]
                for s, gs in enumerate(Glist))
             for j in range(struct_mats.shape[1])]

def x_cond_comp_k_par(k, Glist, Sinv, Mk, struct_mats, K, R, N):
    """
    returns the mean and covariance of ... representing as 
    """

    uks = Uk_repr(Glist, k, struct_mats)

    diagU = [np.diag(u) for u in uks]
    diagU[k] -= Mk

    U = np.column_stack(diagU)

    inv_covar = np.row_stack((
        np.column_stack((np.dot(ui.T, np.dot(Sinv, uj)) for uj in diagU))
        for ui in diagU))

#    inv_covar = np.row_stack((
#        np.column_stack((np.outer(ui, uj)*Sinv for uj in uks))
#        for ui in uks))

    return np.zeros(N*K), inv_covar


"""
Methods related to model setup
"""
def x_gp_init_from_data(obj, **kwargs):
    """
    slightly hacky but allows x_gp_init_from_data to be
    called before data has been passed through fit
    """
    # First make sure there is some data to use
    if hasattr(obj, 'data_Y'):
        Y = obj.data_Y
    elif 'data_Y' in kwargs:
        Y = kwargs['data_Y']
    else:
        msg = "Initalisation of \'x\' from data requires the data to be specified"
        raise AttributeError(msg)

    if obj.aug_latent_vars:
        tt = obj.comp_times
        # lazy interpolation
        obj.x_cur = [np.interp(tt, obj.data_times, y) for y in Y.T]
        
    else:
        # initalise the latent var x at the columns of y
        obj.x_cur = [y for y in Y.T]

def g_gp_init_from_prior(obj):
    if obj.aug_latent_vars:
        tt = obj.comp_times
    else:
        tt = obj.data_times

    obj.g_cur = [gp.sim(tt[:, None], kpar=psi) for gp, psi in zip(obj.g_gps, obj.psi_cur)]


def x_gp_setup(obj, kern_type, kern, kpar):
    if kpar is None:
        kpar = [None for k in range(obj.dim.K)]
    
    if kern_type == 'sqexp':
        x_kernels = [GradientKernel.SquareExponKernel(kp, dim=1)
                     for kp in kpar]

    obj.x_gps = [GradientGaussianProcess(kern) for kern in x_kernels]

    # When setup is called we also initalise current values
    # of the inverse covariance matrices, transformation matrices, etc.

    obj.Cxx_chol = []  # Cholesky factors of the covariance of the 'x' variables
    obj.Cxdx = []      # cross-covar between 'x' and 'dx' for each component
    obj.S_chol = []    # Cholesky decomp of S = Cdxdx|x + gamma[k]*I

    # get the current time values
    if obj.aug_latent_vars:
        tt = obj.comp_times
    else:
        tt = obj.data_times

    # Check to see if gammas has been defined
    # else initalise to default
    if hasattr(obj, 'cur_gammas'):
        gammas = obj.cur_gammas
    else:
        gammas = np.zeros(self.dim.K)
        
    for gamma, gp in zip(gammas, obj.x_gps):
        kern = gp.kernel


def g_gp_setup(obj, kern_type, kern, kpar):
    if kpar is None:
        kpar = [None for k in range(obj.dim.R)]    

    if kern_type == 'sqexp':
        g_kernels = [Kernel.SquareExponKernel(kp)
                     for kp in kpar]
    obj.g_gps = [GaussianProcess(kern) for kern in g_kernels]


##
# creates a sorted full time array from the data
# and augmented times while keeping track of indices
# in the full timeset that correspond to a data point
def sort_augmented_timeset(obj, aug_t):
    data_t = obj.data_times

"""
Adds (or creates) additional latent time variables for the object
"""
def add_latent_states(obj, data_times, aug_times, nspace=1):

    # Create an augmented variable set by adding n_linspace variables
    # in the interval t_i, t_{i+1} for t_i in data_times
    if aug_times == "linspace":
        data_inds = [0]
        comp_times = np.array([data_times[0]])

        for ta, tb in zip(data_times[:-1], data_times[1:]):
            _tt = np.linspace(ta, tb, nspace+2)
            comp_times = np.concatenate((comp_times, _tt[1:]))
            data_inds.append(data_inds[-1] + _tt.size - 1)        
                              
    # User has supplied either a list or numpy array of the augmented
    # time set
    elif isinstance(aug_times, (list, np.ndarray)):
        aug_times = np.array(aug_times)
        comp_times = np.concatenate((data_times, aug_times))
        sort_inds = np.argsort(comp_times)
        comp_times = comp_times[sort_inds]
        data_inds = [np.where(sort_inds == i)[0][0] for i in range(data_times.size)]        

    else:
        err_msg = "aug_times must be of type (list, np.ndarray) or else one of ('linspace')"
        raise ValueError(err_msg)

    obj.data_inds = data_inds
    obj.comp_times = comp_times
    obj.aug_latent_vars = True   

    K = obj.dim.K
    R = obj.dim.R
    N = comp_times.size

    obj.dim = Dimensions(N, K, R)





"""
=====================================================
 Storing of the covariance matrices of the latent
 trajectory 'x' for each independent component as
 well as the conditional covariance of the trajectory
 and its gradient

 also store the matrices 'M' such that E[fdx|fx] = Mx
 for each of the K components
=====================================================
"""
# stores the covariance matrix of the object
# also stores:
# - 
def _store_gpdx_covs(obj):

    obj.Cxx_chol = []  # Cholesky factors of the covariance of the 'x' variables
    obj.Cxdx = []      # cross-covar between 'x' and 'dx' for each component
    obj.S_chol = []    # Cholesky decomp of S = Cdxdx|x + gamma[k]*I

    if obj.aug_latent_vars:
        tt = obj.comp_times[:, None]
    else:
        tt = obj.data_times[:, None]

    gammas = obj._gammas_cur
    In = np.diag(np.ones(obj.dim.N))

    for gp in obj.x_gps:
        kern = gp.kernel

        Cxx = kern.cov(tt, comp='x')
        Cxdx = kern.cov(tt, comp='xdx')
        Cdxdx = kern.cov(tt, comp='dxdx')

        Lxx = la.cholesky(Cxx)

        # attach the relevant covariance matrices

"""
=============================================================
Equations () of is given by
 
 ___            T   -1                     -1
 | | exp{ -0.5*x  Cxk  x -0.5 (fk - Mk x) S  (fk - Mk x) }
 k=1                                 

where
 - Cxk is the prior covariance matrix of the kth latent GP
 - Mk = Cov[ ]
 
=============================================================
"""
def gpdx(tt, kern, kpar=None):
    """
    I return some values from :eq:`myeq`

    :param tt: array of input times
    :param kern: kernel object
    :param kpar: option parameters for the kernel
    :rtype: three matrics
    """
    Cxx = kern.cov(tt, comp='x')
    Cxdx = kern.cov(tt, comp='xdx')
    Cdxdx = kern.cov(tt, comp='dxdx')


def dx_gp_condmats(tt, kern, gamma, kpar=None, Cxx_return=None):
    """
    reference to :class:`~pydygp.kernels.GradientKernel`
    """
    I = np.eye(tt.size) 
    Cxx = kern.cov(tt[:, None], kpar=kpar)
    Cxdx = kern.cov(tt[:, None], kpar=kpar, comp='xdx')
    Cdxdx = kern.cov(tt[:, None], kpar=kpar, comp='dxdx')

    # Get the cholesky decomp. of Cxx
    Cxx_chol = la.cholesky(Cxx)

    # Covariance matrix of dfdx | x for the GP with kernel `kern`
    Cdxdx_x = Cdxdx - np.dot(Cxdx.T, la.back_sub(Cxx_chol, Cxdx))

    # The model involves adding a small noise term to Cdxdx_x
    S = Cdxdx_x + gamma**2*I

    # Take the cholesky decomposition of S
    S_chol = la.cholesky(S)

    # Get the transformation matrix M such that Mx = E[dfdx | X=x]
    M = np.dot(Cxdx.T, la.back_sub(Cxx_chol, I))

    if Cxx_return is None:
        return M, S_chol

    elif Cxx_return == 'chol':
        # Also return the cholesky decomp. of Cxx
        # rather than doing it elsewhere
        return M, S_chol, Cxx_chol

"""
Methods related to model initialisation
"""
def init_latent_vars(obj, x0, g0s):

    """
    Trajectory initalisation
    """
    # User supplied initial value
    if isinstance(x0, np.ndarray):
        obj.mh_pars.xcur = (x0, None)

    # Set inital latent variables equal to state
    elif x0 == 'data':
        assert(not obj.aug_latent_vars)
        obj.mj_pars.xcur = (obj.data_Y.copy(), None)

    # optional fit gp to observed data and then use the fitted
    # process to estimate latent states
    elif x0 == 'gpfit':
        pass 


    """
    Latent force initalisation
    """

    # User supplied inital force values
    if isinstance(g0s, list):
        for r in range(obj.dim.R):
            obj.mh_pars.gcur[r] = (g0s[r], None)

    elif g0s == 'prior':
        tt = obj.data_times[:, None]
        for r, gp in enumerate(obj.g_gps):
            gp.fit(tt)
            rv = gp.sim()
            obj.mh_pars.gcur[r] = (rv, None)
