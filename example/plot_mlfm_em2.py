# -*- coding: utf-8 -*-
"""
EM Fit of the MLFM
==================

Uses the EM algorithm to fit estimates of the latent forces for the model

   .. math::

      \dot{\mathbf{x}(t)} = \\left(\mathbf{L}_x + \mathbf{L}_y g_1(t) + \mathbf{L}_z g_2(t) \\right)\mathbf{x}(t)

where :math:`g_1(t)` and :math:`g_2(t)` are independent Gaussian processes, and :math:`\mathbf{L}_i` are the
standard basis of the Lie algebra :math:`\mathfrak{so}(3)` of the rotation group :math:`SO(3)`.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from pydygp.gaussianprocesses import GaussianProcess
from pydygp.linlatentforcemodels import MLFM

def main():

    np.random.seed(122)

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

    # make the basic mlfm model
    mlfm = MLFM.ns([Lx, Ly, Lz], order=1)

    # latent force gaussian processes
    g_gps = [GaussianProcess('sqexp', kpar=[1., 1.])
             for r in range(len(mlfm.struct_mats)-1)]

    # simulate some data 
    T = 4.
    Nd = 15
    
    tt = np.linspace(0., T, Nd)

    # dense set of times for solving the ODE
    tt_d = np.linspace(0., T, 100)

    # set of times for simulating the latent trajectory
    tt_gp_sim = np.linspace(tt[0], tt[-1], 25)

    y, y_dense, g, _ = mlfm.sim([0., 0., 1.],
                                tt,
                                gps=g_gps,
                                tt_dense=tt_d,
                                tt_gp_sim=tt_gp_sim,
                                return_gp=True)    

    # inital gaussian process approx
    #  - ind. gps, for each dimension
    x0_gps = [GaussianProcess('sqexp', kpar=[1., .7]) for k in range(3)]

    fig = plt.figure()
    ax1 = plt.subplot2grid((1, 2), (0, 0), colspan=1)
    ax2 = plt.subplot2grid((1, 2), (0, 1), colspan=1)

    for M in [10]:

        # reduced inds
        red_ind = np.linspace(0, tt.size-1, Nd, dtype=np.intp)
        red_tt = tt[red_ind]
        red_y = y[red_ind, :]

        mlfm.order = M

        # sets up the em fit
        mlfm.em_fit(red_tt,
                    red_y.T.ravel(),
                    x0_gps=x0_gps, g_gps=g_gps,
                    ifix=Nd//2-1,
                    h=None)
        
        mlfm.em.cov_setup()

#        ghat = np.random.normal(size=mlfm.em.dim.N*mlfm.em.dim.R)
#        m1, c1, ccross = mlfm.em.Estep_LDS(ghat)
#        m2, c2 = mlfm.em.Estep(ghat)

#        g1 = mlfm.em.Mstep(z_mean=m1, covz=c1, covzzp=ccross)
#        g2 = mlfm.em.Mstep(z_mean=m2, z_cov=c2)

        ghat = mlfm.em.fit(liktol=5e-2, max_nt=1000).reshape(mlfm.em.dim.R, mlfm.em.dim.N)
  
if __name__ == '__main__':
    main()    

    

    
