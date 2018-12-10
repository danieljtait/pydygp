import numpy as np
import scipy.sparse as sparse

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

def T_xrav_tovecx(N, M):
    """ Returns the NM x NM sparse matrix mapping X.ravel() to vec(X)
    """
    T = []
    for n in range(N*M):
        # image of a basis vector
        c = n - (n // M)*M   # col.
        r = n // M           # row
        # transpose
        c, r = (r, c)

        Ten = sparse.lil_matrix((N*M, 1))
        Ten[r*N + c] = 1.
        T.append(Ten)

    return sparse.hstack(T)

def _get_gp_theta_shape(gp_list):
    """ Handler function for getting the shape of the
    free hyper parameters for the a list of Gaussian Processes objects
    """
    return [gp.kernel.theta.size for gp in gp_list]
