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
    def xgrad_cond_covs(self):
        """
        The conditional covariances 'dx|x'
        """
        try:
            return self._xgrad_cond_covs
        except:
            covs = gpdx_cond__cov(self.ttf, self.x_gps)
            self._xgrad_cond_covs = covs
            return covs

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
    Model Utility Functions
    """
    def _x_flow_rep(self, k, vecg):
        col_g = vecg.reshape(self.dim.R, self.dim.N).T
        return x_flow_rep(k, self.struct_mats, col_g, self.dim.K)

def gpdx_cond__cov(tt, gps):
    """
    Returns the set of conditional covariance matrices
    of the gradients
    """
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

        Cdx_x_list.append(Cdx_x)

    return Cdx_x_list


def x_flow_rep(k, A, g_colform, K):
    """
    Returns the set [u_k1,...,ukK] where
    u_kj = A[0, k, j] + sum_r A[r, k, j]*g_colform[:, r]
    """
    return [A[0, k, j] + sum([Ar[k, j]*gr
                              for Ar, gr in zip(A[1:,],
                                                g_colform.T)])
            for j in range(K)]


def x_cond_g_par(x, gcol, A,
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
        uk = x_flow_rep(k, A, gcol, K)

        # the transformation matrix E[dxk|x] = Mk x
        Mk = Mk_list[k]

        # matrix Cdx|x + gamma_k^2I
        Skinv = Skinv_list[k]

        ic = np.zeros((NK, NK))
        for i in range(K):
            for j in range(i):
                uki_ukjT = np.outer(uk[i], uk[j])
                ic[i*NK:(i+1)*NK, j*NK:(j+1)*NK] += uki_ukjT

                # Add the term Mk.T Skinv d(ukj) to the kth row
                if i==k:
                    ic[i*NK:(i+1)*NK, j*NK:(j+1)*NK] += np.dot(Mk.T, np.dot(Skinv, np.diag(uk[j])))

            # scale for transpose sum
            ic[i*NK:(i+1)*NK, i*NK:(i+1)*NK] = 0.5*np.outer(uk[i], uk[i])

        # add the term:  Mk.T Skinv Mk
        ic[k*NK:(k+1)*NK, k*NK:(k+1)*NK] = 0.5*np.dot(Mk.T, np.dot(Skinv, Mk))

        ic = ic + ic.T

        # add the result to inv_cov
        inv_cov.append(ic)            

    ic = sum(inv_cov)
