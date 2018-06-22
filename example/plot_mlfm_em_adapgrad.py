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
    mlfm = MLFM().adapgrad([Lx, Ly, Lz])

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

    k = 0
    for u in mlfm.em._x_flow_rep(k, gg):
        print(u)
    print("----------")
    for j in range(3):
        print(Lx[k, j] + Ly[k, j]*np.sin(tt) + Lz[k, j]*np.cos(tt))
                        

if __name__ == '__main__':
    main()



