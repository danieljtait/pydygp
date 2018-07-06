"""
================================
MLFM MAP Latent Force Estimation
================================

Demonstrates the use of the adaptive gradient
matching method to carry out maximum a posteriori
estimates of the latent force functions.
"""
import numpy as np
from pydygp.kernels import GradientKernel
from pydygp.gaussianprocesses import GaussianProcess, GradientGaussianProcess
from pydygp.linlatentforcemodels import MLFM
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)


def main():
    """
    Data simulation
    """
    # infinitesimal rotation matrices in so(3)
    Lx = np.array([[0., 0., 0.],
                   [0., 0.,-1.],
                   [0., 1., 0.]])

    Ly = np.array([[ 0., 0., 1.],
                   [ 0., 0., 0.],
                   [-1., 0., 0.]])

    Lz = np.array([[ 0.,-1., 0.],
                   [ 1., 0., 0.],
                   [ 0., 0., 0.]])

    # make the basic mlfm model, using the adaptive gradient matching
    mlfm = MLFM.adapgrad([Lx, Ly, Lz])

    # latent force gaussian processes
    g_gps = [GaussianProcess('sqexp', kpar=[1., 1.])
             for r in range(len(mlfm.struct_mats)-1)]

    """
    Setup the adaptive gradient matching model
    """
    # Create some the GP interpolator of the state trajectory
    x_kernel_hyperpars = ([1.0, 1.0],
                          [1.0, 0.9],
                          [1.1, 1.0], )
    x_kernels = [GradientKernel.SquareExponKernel(par, dim=1) for par in x_kernel_hyperpars]
    x_gps = [GradientGaussianProcess(kern) for kern in x_kernels]

    # setup the EM method
    mlfm.em_fit(x_gps, g_gps)

    tt = np.linspace(0., 3., 3)
    mlfm.em.time_input_setup(tt)
    mlfm.em.gammas = 0.1*np.ones(mlfm.em.dim.K)

    gg = np.column_stack((np.sin(tt), np.cos(tt))).T.ravel()
    mlfm.em._update_x_cov_structure()


    #### Test functions
    def f(x, t):
        return np.dot(Lx + np.sin(t)*Ly + np.cos(t)*Lz, x)

    def mk(x, k):
        return np.dot(mlfm.em.Mdx[k], x)

    def ll(x):
        X = x.reshape(mlfm.em.dim.K, mlfm.em.dim.N).T
        F = np.row_stack([f(x, t) for x, t in zip(X, tt)])

        exp_arg = 0.
        for k in range(mlfm.em.dim.K):
            etak = F[:, k] - mk(X[:, k], k)
            exp_arg += -0.5*np.dot(etak, np.dot(mlfm.em.Sinv_covs[k], etak))

        return exp_arg

    x = np.zeros(9) + 1e-3
    from scipy.optimize import minimize
    res = minimize(lambda x: -ll(x), x)
    Hinv = res.hess_inv
    print(np.diag(Hinv))
    print(np.diag(np.linalg.inv(Hinv)))
    
    mlfm.em.Estep(gg)
    
if __name__ == '__main__':
    main()



