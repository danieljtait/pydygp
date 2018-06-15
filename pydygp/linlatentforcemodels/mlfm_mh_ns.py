# External imports
from collections import namedtuple
import numpy as np
from scipy.linalg import block_diag
from scipy.stats import gamma

# Package imports
from pydygp.gaussianprocesses import GaussianProcess
from pydygp.kernels import Kernel

# Local imports
from pydygp import nssolve

Dimensions = namedtuple('Dimensions', 'N K R')

class MLFM_MH_NS:

    def __init__(self, structure_matrices):

        self.struct_mats = np.asarray(structure_matrices)
        self.dim = Dimensions(N=None,
                              K=self.struct_mats.shape[1],
                              R=self.struct_mats.shape[0]-1)

        # various model flags
        self.is_phi_inv_scale = True  # For each k, 1/phi_k[0] follows a Gamma(a, b) (default)
        self.is_psi_inv_scale = True  # For each k, 1/psi_k[0] follows a Gamma(a, b) (default)
        
    
    def operator_setup(self):
        """
        Sets up the NS quadrature operator
        """
        pass

    def time_interval_setup(self, data_times, h=None):
        """
        Sets up the intervals and handles the number of time points in
        each interval
        """
        intervals = [nssolve.ns_util.Interval(ta, tb)
                     for ta, tb in zip(data_times[:-1], data_times[1:])]
        data_inds = [0]
        for I in intervals:
            I.set_quad_style(h=h)
            data_inds.append(data_inds[-1]+I.tt.size-1)
            
        self.intervals = intervals
        self.comp_times = nssolve.ns_util.get_times(self.intervals)
        self.data_inds = data_inds
        self.dim = Dimensions(self.comp_times.size, self.dim.K, self.dim.R)
        

    def phi_setup(self, phi_inv_scale_abs=None, phi_priors=None, phi_proposals=None):
        """
        Sets up the x gp hyperparameters
        
        Allows the option to specify a prior and proposal distribution for the
        hyper parameters, :math:`\mathbf{\phi}_k` for each of the :math:`K` Gaussian
        processes modelling the latent trajectories

        The model also assumes the :math:`\phi_{k0}` parameter is a 'scale' parameter
        and therefore its inverse may be given a conjugate Gamma(a, b) prior. This
        defauls to a Gamma(1, 1) prior for each dimension although may be customised
        by providing a list of length K
        """
        if phi_inv_scale_abs is None:
            self.phi_inv_scale_abs = [(1, 1)]*self.dim.K
        self.phi_priors = phi_priors
        self.phi_proposals = phi_proposals

    def psi_setup(self, psi_inv_scale_abs=None, psi_priors=None, psi_proposals=None):
        pass
    

    def x_gp_setup(self,
                   kern_type='sqexp',
                   x_kernels=None):
        if x_kernels is not None:
            # do some checks on what has been supplied
            assert(len(x_kernels) == self.dim.K)

        elif x_kernels is None:
            if kern_type == 'sqexp':
                x_kernels = [Kernel.SquareExponKernel(None, dim=1)
                             for k in range(self.dim.K)]
            else:
                raise ValueError('Unrecognised kernel type')

        self.x_gps = [GaussianProcess(kern) for kern in x_kernels]


    def g_gp_setup(self, kern_type='sqexp', g_kernels=None):
        g_kernels = [Kernel.SquareExponKernel(None, dim=1)
                     for k in range(self.dim.R)]
        self.g_gps = [GaussianProcess(kern) for kern in g_kernels]

    def Lambda_setup(self, scale, df):
        """
        
        """
        self.Lambda_scale = scale
        self.Lambda_df = df



    """
    Initialisation functions
    """
    def variables_init(self):
        """
        Initalises the model after setup has been completed.
        """
        for var in ('phi', 'psi', 'x', 'g'):
            if getattr(self, "is_{}_init".format(var) ):
                var_init_method = getattr(self, "{}_init".format(var))
                var_init_method()

    
    def phi_init(self, phi_val=None, prior_rv=False):
        if prior_rv:
            # make sure the prior methods have a random variable method
            if all(p.rvs is None for p in self.phi_priors):
                raise ValueError("Prior.rvs method must be specified for initalisation \"prior_rv\"")
            
            self.phi_cur = [p.rvs() for p in self.phi_priors]

        elif phi_val is not None:
            assert(len(phi_val) == self.dim.K)
            self.phi_cur = phi_val

        # Add catches
        self.is_phi_init = True

    """
    Conditional distribution functions
    """
    def x_cond_meanvar(self):

        tt = self.comp_times

        # get the prior covariance for each component
        Cx = [gp.kernel.cov(tt[:, None], kpar=phi) for gp, phi in zip(self.x_gps,
                                                        self.phi_cur)]
        for c in Cx:
            c += np.diag(1e-3*np.ones(c.shape[0]))
        Cx_inv = [np.linalg.inv(c) for c in Cx]

        Cx_inv = block_diag(*Cx_inv)

        K = self.myop.x_transform(np.concatenate(self.g_cur),
                                  is_x_input_vec=True,
                                  is_x_output_vec=True)

        vec_y = self.data_Y.T.ravel()

        A = np.row_stack([np.eye(1, M=self.dim.N, k=ind) for ind in self.data_inds])
        A = block_diag(*[A]*self.dim.K)
        AK = np.dot(A, K)
        inv_cov = Cx_inv + np.dot(AK.T, np.dot(self.Lambda_cur, AK))
        pre_mean = np.dot(K.T, np.dot(A.T, np.dot(self.Lambda_cur, vec_y)))
        cov = np.linalg.inv(inv_cov)


        mean = np.linalg.solve(inv_cov, pre_mean)
        Kmean = np.dot(K, mean)
        return mean, cov

    def g_cond_meanvar(self, jitter=True):

        tt = self.comp_times

        # get the prior covariance for each latent force
        Cg = [gp.kernel.cov(tt[:, None], kpar=psi)
              for gp, psi in zip(self.g_gps, self.psi_cur)]

        # Optional add jitter matrix for stability
        if jitter:
            for c in Cg:
                c += np.diag(1e-5*np.ones(c.shape[0]))
            
        Cg_inv = [np.linalg.inv(c) for c in Cg]
        Cg_inv = block_diag(*Cg_inv)

        X = np.column_stack(self.x_cur)
        L, b = self.myop.g_transform(X.ravel(), is_x_input_vec=False, is_x_output_vec=True)

        vec_y = self.data_Y.T.ravel()

        A = np.row_stack([np.eye(1, M=self.dim.N, k=ind)
                          for ind in self.data_inds])
        A = block_diag(*[A]*self.dim.K)

        eta = vec_y - np.dot(A, b)
        AL = np.dot(A, L)

        inv_cov = Cg_inv + np.dot(AL.T, np.dot(self.Lambda_cur, AL))
        pre_mean = np.dot(L.T, np.dot(A.T, np.dot(self.Lambda_cur, eta)))

        mean = np.linalg.solve(inv_cov, pre_mean)
        cov = np.linalg.inv(inv_cov)

        return mean, cov

    def gibbs_inv_phi_scale_ab(self):
        """
        Returns a list of parameters a, b corresponding to the
        posterior of the scale parameter of the absolute scale
        parameter of the x gp covariance matrices.

        Each of the parameters :math:`phi_{k0}` is an absolute
        scale constant for the covariance matrix of the trajectory
        Gaussian processes and so the precision may be given a
        conjugate gamma prior
        """

        assert(self.is_phi_inv_scale)
        
        alist = []
        blist = []
        for k, x_gp in enumerate(self.x_gps):

            xk = self.x_curs[k]

            a0, b0 = self.phi_inv_scale_abs[k]
            aN = a0 + self.dim.N
            bN = b0 + 0.5*np.dot(xk, np.dot(x_gp.inv_cov, xk))

            alist.append(aN)
            blist.append(bN)

        return alist, blist


"""
Initalisation Utility functions
-------------------------------
"""
def x_init_from_data(obj, **kwargs):
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
