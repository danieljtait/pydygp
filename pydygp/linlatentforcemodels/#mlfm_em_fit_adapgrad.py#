import numpy as np
from collections import namedtuple

# Model dimensions:
#   N - size of augmented time vector
#   K - dimension of the ambient space
#   R - number of latent forces
Dimensions = namedtuple('Dimensions', 'N K R')

class MLFM_AdapGrad_EM:
    """
    Carries out EM estimation of the latent forces for the
    MLFM using the adaptive gradient method.
    """
    def __init__(self, struct_mats, x_gps, g_gps):

        self.x_gps = x_gps
        self.g_gps = g_gps

        # store the struct_mats
        self.struct_mats = struct_mats

        # dimensions
        self.dim = Dimensions(None, struct_mats[0].shape[0], len(struct_mats)-1)

        # flags
        self.is_comp_data = False

    @property
    def ttf(self):
        """
        The full time vector, equal to the data times if the model
        has not been completed.
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
            return self._tt_data
        except:
            return None

    @property
    def x_cov_chols(self):
        """
        List of the cholesky decomposition of the covariance
        matrices of the states xk
        """
        try:
            return self._x_cov_chols
        except:
            raise ValueError

    @property
    def xdx_cross_covs(self):
        """
        List of the cross covariance between the latent
        forces and the states
        """
        try:
            return self._xdx_cross_covs
        except:
            raise ValueError

    @property
    def xgrad_cond_covs(self):
        """
        The conditional covariances 'dx|x'
        """
        try:
            return self._xgrad_cond_covs
        except:
            # for now we are only updating these values
            # through _update_x_cov_structure
            raise ValueError

    @property
    def Sinv_covs(self):
        """
        Inverses of the matrices Sk  given by Cdx|x + gammmas[k]**2*I

        .. math::
           S_k = C_{\dot{x}\dot{x}} - C_{\dot{x}x}C_{xx}^{-1}C_{x\dot{x}} + \gamma_k^2 I
        """
        try:
            return self._Sinv_covs
        except:
            S_list = [c + g**2*np.eye(self.dim.N)
                      for c, g in zip(self.xgrad_cond_covs,
                                      self.gammas)]
            self._Sinv_covs = [np.linalg.inv(S) for S in S_list]
            return self._Sinv_covs

    @property
    def Mdx(self):
        """
        Transformation matrices Mk such that E[dx|x] = [Mk]x
        """
        try:
            return self._Mdx_list
        except:
            raise ValueError

    """
    Model setup functions
    *********************
    """
    def time_input_setup(self, data_times, comp_times=None, data_inds=None):
        """
        The model can be augmented by including additional latent times
        """
        self._tt_data = data_times
        self._comp_times = comp_times
        self.data_inds = data_inds

        if comp_times is not None:
            N = comp_times.size
        else:
            N = data_times.size

        # Update dimension
        self.dim = Dimensions(N, self.dim.K, self.dim.R)
        
    def x_gp_setup(self, x_gps):
        # make sure these all gradient gps
        pass


    """
    EM fit functions
    ****************
    """
    def Estep(self, g):
        """
        Returns the mean and covariance of the latent states
        conditional on the latent force g
        """
        gcol = g.reshape(self.dim.R, self.dim.N).T
        ic = x_cond_g_par(gcol, self.struct_mats,
                          self.Mdx, self.Sinv_covs,
                          self.dim.N, self.dim.K)
        c = np.linalg.inv(ic)
        print(np.diag(ic))

    """
    Model Utility Functions
    """
    def _x_flow_rep(self, k, vecg):
        col_g = vecg.reshape(self.dim.R, self.dim.N).T
        return x_flow_rep(k, self.struct_mats, col_g, self.dim.K)

    def _update_x_cov_structure(self):
        """
        Updates
        * the cholesky decomposition of the latent states
        * the cross covariance C(x,dx) of the states and gradient
        * the gradient conditional cov matrices
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

"""
General utility and helper functions for the EM
fit of the MLFM adapgrad model
"""

def gpdx_cond_cov(tt, gps):
    """
    Returns the set of conditional covariance matrices
    of the gradients
    """
    Lxx_list = []
    Cxdx_list = []
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


def x_flow_rep(k, A, g_colform, K):
    """
    Returns the set [u_k1,...,ukK] where
    u_kj = A[0, k, j] + sum_r A[r, k, j]*g_colform[:, r]
    """
    return [A[0, k, j] + sum([Ar[k, j]*gr
                              for Ar, gr in zip(A[1:,],
                                                g_colform.T)])
            for j in range(K)]

def _get_Wi(G, i, A):
    Ws_i = [sum(A[s+1, i, j]*gs + A[0, i, j]
                for s, gs in enumerate(G))
             for j in range(A.shape[1])]
    return Ws_i

def x_cond_g_par(gcol, A,
                 Mk_list, Skinv_list,
                 N, K):
    """
    Returns the inv. covariance matrix of the Gaussian conditional p(x|g)
    from the adap. grad. model
    """
    # dimensions
    NK = N*K
    
    inv_cov = []
    for k in range(K):

        # get the component vectors of the flow representation
        #uk = x_flow_rep(k, A, gcol, K) <-------- !!!!! something going wrong here
        uk = _get_Wi(gcol.T, k, A)

        # the transformation matrix E[dxk|x] = Mk x
        Mk = Mk_list[k]

        # matrix Cdx|x + gamma_k^2I
        Skinv = Skinv_list[k]

        diagUk = [np.diag(uki) for uki in uk]
        diagUk[k] -= Mk

        ic = np.row_stack((
            np.column_stack((np.dot(uki.T, np.dot(Skinv, ukj)) for ukj in diagUk))
            for uki in diagUk))
        inv_cov.append(ic)        
    ic = sum(inv_cov)
    ic += np.diag(1e-4*np.ones(NK))
    return ic


def parse_component_k_for_x(k, G, Skinv, Mk, A,
                            K, R, N):

    uk = _get_Wi(G, k, A)
    diagUk = [np.diag(uki) for uki in uk]

