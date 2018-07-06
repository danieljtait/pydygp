import numpy as np
from pydygp.kernels import Kernel, GradientKernel
from pydygp.gaussianprocesses import GaussianProcess
from collections import namedtuple
from scipy.linalg import solve_triangular
# Default settings

X_KERN_DEFAULT = 'sqexp'
GAMMAS_DEFAULT = 1e-1

# Model dimensions:
#   N - size of augmented time vector
#   K - dimension of the ambient space
#   R - number of latent forces
Dimensions = namedtuple('Dimensions', 'N K R')



class MLFM_AdapGrad:
    """Base class for MLFM with adaptive gradient matching.

    """
    def __init__(self, struct_mats):
        self.struct_mats = np.asarray(struct_mats)
        self.dim = Dimensions(None, self.struct_mats.shape[-1], self.struct_mats.shape[0]-1)
        
        # various model flags
        self.is_comp_data = False
        self.data_precision_is_cong = True

    def x_flowk_rep(self, k, vecg):
        """
        Represents the flow vector fk as sum_j ukj o xj
        """
        A = self.struct_mats
        G = vecg.reshape(self.dim.R, self.dim.N)
        uk = [A[0, k, j] + sum([ar[k, j]*gr for ar, gr in zip(A[1:,:,:], G)])
              for j in range(self.dim.K)]
        return uk

    def g_flowk_rep(self, k, vecx):
        """
        Represents the flow vector fk as sum_r vkr o gr
        """
        A = self.struct_mats
        X = vecx.reshape(self.dim.K, self.dim.N)
        # contrib. from the offset matrix        
        vk0 = sum(A[0, k, j]*xj for j, xj in enumerate(X))  
        vk = [sum(ar[k, j]*xj for j, xj in enumerate(X)) for ar in A[1:]]
        return vk, vk0


    def setup(self, data_times, data_Y=None, aug_times=None, **kwargs):
        """
        default setup function carries out model initalisation
        """
        # Attachement of data and times
        self._data_times = data_times
        self.data_Y = data_Y

        # Update dimensions to add number of data points
        self.dim = Dimensions(data_times.size, self.dim.K, self.dim.R)

        # setup of the model variables called in a way respecting
        # the model hierarchy
        self._gammas_setup()
        #self._xp_setup(aug_times, **kwargs)

    def setup_latentforce_gps(self, kernels):
        self.g_gps = [GaussianProcess(k) for k in kernels]

    def setup_data(self,
                   data_times, data_Y,
                   comp_times=None, data_inds=None):
        """Attach data to the model, optionally include augmented latent state points
        """
        if comp_times is None:
            # no additional latent states
            self._data_times = data_times
        else:
            if data_inds is not None:
                self._comp_times = comp_times
            else:
                raise ValueError("Must supply the indices of data obs. in the completed time set")
                

    """
    Setup prior and proposal functions
    * data precisions
    * latent state gp hyperparameters
    * latent force gp hyperparameters
    * prior for offset matrix (ToDo)
    """
    def setup_data_precision(self, ab=None):
        """Prior and proposal distribution for the data precisions - defaults to
        conjugate gamma
        """
        if ab is not None:
            # setup is being done using the conjugate gamma 
            self.data_precision_ab = ab
        else:
            self.data_precision_is_cong = False            
            raise NotImplementedError('Currently only the conjugate gamma is supported')


    def _gammas_setup(self, gammas=None):
        if gammas == None:
            self.gammas = GAMMAS_DEFAULT*np.ones(self.dim.K)

        else:
            self.gammas = gammas

    def _x_gp_setup(self, aug_times=None, x_kern='sqexp', x_kpar=None, **kwargs):
        """ Sets up the latent gp kernels
        """

        # enhances the model with additional latent input times
        if aug_times is not None:
            add_latent_states(self, self.data_times, aug_times, **kwargs)

    def x_cond_posterior(self, x, Y=None, gs=None, x_kpars=None):
        """ Evaluates the conditional posterior

        .. math::
           p(\mathbf{x}|\mathbf{y}, \mathbf{g}, \\boldsymbol{\phi})

        [more explanation]
        """
        if Y is None:
            Y = self.data_Y

        if x_kpars is None:
            x_kpars = [None for k in range(self.dim.K)]

        fks = None

        logp = 0.

        for k, gp in enumerate(self.x_gps):
            
            Mk, Sk_chol, Cxx_chol = dx_gp_condmats(tt, kern, x_kpars[k], 'chol')


    """
    Model Utility Functions
    """
    def _update_x_cov_structure(self):
        """
        Updates
        * the cholesky decomposition of the latent states
        * the cross covariance C(x,dx) of the states and their gradients
        * the gradient conditioned on state cov matrices
        """
        Lxx, Cxdx, Cdx_x = gpdx_cond_cov(self.ttf, self.x_gps)
        self._x_cov_chols = Lxx
        self._xdx_cross_covs = Cxdx
        self._x_grad_cond_covs = Cdx_x

        Mk_list = [np.dot(cxdx.T,
                          np.linalg.solve(L.T,
                                          np.linalg.solve(L, np.eye(L.shape[0]))))
                   for cxdx, L in zip(self._xdx_cross_covs, self._x_cov_chols)]
        self._Mdx_list = Mk_list

        Sinv_list = [np.linalg.inv(c+g**2*np.eye(c.shape[0]))
                     for c, g in zip(Cdx_x, self.gammas)]
        self._Sinv_covs = Sinv_list    

    def _update_g_cov_structure(self):
        # add a bit of alpha noise
        aI = 1e-5*np.eye(self.dim.N)
        Cgg = [gp.kernel.cov(self.ttf[:, None]) + aI for gp in self.g_gps]

        # calculate the cholesky decomp.
        self._g_cov_chols = [np.linalg.cholesky(c) for c in Cgg]
        

    def log_model_likelihood(self, vecx, vecg,
                             Mdx_mats=None, Skinv_covs=None):
        """Loglikelihood of the ODE model dependent part of the model
        """
        ell = 0.

        # bit of reshaping
        X = vecx.reshape(self.dim.K, self.dim.N)

        if Mdx_mats is None:
            Mdx_mats = self._Mdx_list
        if Skinv_covs is None:
            Skinv_list = self._Sinv_covs
        
        for k in range(self.dim.K):

            uk = self.x_flowk_rep(k, vecg)

            # kth component of the flow function 
            fk = sum([ukj*xj for ukj, xj in zip(uk, X)])

            # conditioned estimated of the flow
            mk = np.dot(Mdx_mats[k], X[k, :])

            etak = fk - mk

            ell += -0.5*np.dot(etak, np.dot(Sinv_covs[k], etak))

        return ell

    def model_x_dist(self, vecg):
        """
        Conditional on g the model can be rearranged as the quad. form of
        a normal in X
        """
        inv_cov = []
        for k in range(self.dim.K):
            uk = self.x_flowk_rep(k, vecg)
            Mk = self._Mdx_list[k]
            Skinv = self._Sinv_covs[k]

            diagUk = [np.diag(uki) for uki in uk]
            diagUk[k] -= Mk

            ic = np.dot(np.row_stack([dki.T for dki in diagUk]),
                        np.column_stack([np.dot(Skinv, dkj) for dkj in diagUk]))

            inv_cov.append(ic)

        ic = sum(inv_cov)
        return ic
        
    def log_prior(self, vecx=None, vecg=None):
        """logpdf of the prior 
        """
        ell = 0.

        if vecx is not None:
            X = vecx.reshape(self.dim.K, self.dim.N)  # reshape

            for x, L in zip(X, self._x_cov_chols):
                ell += -0.5*np.dot(x,
                                   solve_triangular(L.T,
                                                    solve_triangular(L, x, lower=True)))

        if vecg is not None:
            G = vecg.reshape(self.dim.R, self.dim.N)  # reshape

            for g, L in zip(G, self._g_cov_chols):
                ell += -0.5*np.dot(g,
                                   solve_triangular(L.T,
                                                    solve_triangular(L, g, lower=True)))


        return ell

    @property
    def ttf(self):
        """The full time vector

        equal to the data times if no additional latent states included
        """
        if self.is_comp_data:
            return self._comp_times
        else:
            return self.data_times


    @property
    def data_times(self):
        """
        Time points for attached data
        """
        try:
            return self._data_times
        except:
            return None


    @property
    def data_map(self):
        """
        Matrix that maps the augmented latent states
        to the data for which there is an observed data
        point
        """
        data_map = np.row_stack((np.eye(N=1, M=self.dim.N, k=i)
                                 for i in self.data_inds))
        data_map = block_diag(*[data_map]*self.dim.K)
        return data_map        


"""
General utility and helper functions for the mlfm
adaptive gradient matching methods
"""

def gpdx_cond_cov(tt, gps):
    """
    Returns the set of conditional cov. matrices of the gradients
    """
    Lxx_list = []   # Cholesky decomposition of state cov
    Cxdx_list = []  #
    Cdx_x_list = []

    for gp in gps:

        # auto cov. of the state
        Cxx = gp.kernel.cov(tt[:, None], comp='x')
        # cross cov. of the state and grad
        Cxdx = gp.kernel.cov(tt[:, None], comp='xdx')
        # auto cov. of the grad
        Cdxdx = gp.kernel.cov(tt[:, None], comp='dxdx')

        Lx = np.linalg.cholesky(Cxx)

        # cond. cov of the grad given the state
        Cdx_x = Cdxdx - np.dot(Cxdx.T, np.linalg.solve(Lx.T,
                                                       np.linalg.solve(Lx, Cxdx)))

        Lxx_list.append(Lx)
        Cxdx_list.append(Cxdx)
        Cdx_x_list.append(Cdx_x)

    return Lxx_list, Cxdx_list, Cdx_x_list


def gpdxk_cond_cov(tt, gpk, kpar=None, alpha=0):
    """
    Returns the set of conditional cov. matrices of the gradients
    """
    Lxx_list = []   # Cholesky decomposition of state cov
    Cxdx_list = []  #
    Cdx_x_list = []

    Cxx = gpk.kernel.cov(tt[:, None], comp='x')
    Cxdx = gpk.kernel.cov(tt[:, None], comp='xdx')
    Cdxdx = gpk.kernel.cov(tt[:, None], comp='dxdx')

    if alpha > 0:
        Cxx += np.diag(alpha*np.ones(tt.size))
    
    Lxx = np.linalg.cholesky(Cxx)

    Cdx_x = Cdxdx - np.dot(Cxdx.T, np.linalg.solve(Lx.T,
                                                   np.linalg.solve(Lx, Cxdx)))

    return Lxx, Cxdx, Cdx_x

        
    
class MLFMAdapGrad_MCMC(MLFM_AdapGrad):

    @property
    def cur_data_precision(self):
        # current MC sample of the data precision
        return self._cur_data_precision
    
    """MLFM with adaptive gradient matching and model fitting done
    using MCMC methods
    """
    def __init__(self, *args, **kwargs):
        super(MLFMAdapGrad_MCMC, self).__init__(*args, **kwargs)

    """
    Variable initalisation functions for MCMC fitting
    """
    def init_data_precision(self, strategy='value', value=None):

        if strategy == 'value':
            # data precision terms initalised by value 
            assert(value is not None)

        elif strategy == 'rvs':
            # initalise by drawing from prior
            pass
            
    def data_precision_update(self):
        if self.data_precision_is_cong:
            # Update done using the conjugate gamma distribution
            pass


    def xk_hyperpar_mh_update(self):

        # rewrite the conditional log pdf for only those terms
        # dependent on the hyperparameter of xk
        def lk(phi_k):
            Lxx, Mdx, Cdx_x = gpdxk_cond_cov(self.ttf,
                                             self.x_gps[k],
                                             kpar=phi_k)
                                             
