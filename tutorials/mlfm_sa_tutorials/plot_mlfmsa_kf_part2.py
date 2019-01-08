"""

Recovering the Latent Force
===========================

The :ref:`previous note <tutorials-mlfmsa-motiv-part1>` demonstrated that it
is possible to recover the latent by inverting the trajectories formed by
the operator evaluated at the known true values. This is obviously of
limited use and so in this note we will expand this construction to an
iterative procedure for estimating the unknown forces.

We set up the model and simulate the data exactly as was done previously

"""
import numpy as np
from pydygp.liealgebras import so
from sklearn.gaussian_process.kernels import RBF
from pydygp.linlatentforcemodels import MLFMSA

mlfm = MLFMSA(so(3), R=1, lf_kernels=[RBF(), ], order=10)
beta = np.array([[0.1, 0., 0.],
                 [-0.5, 0.31, 0.11]])
t1 = np.linspace(0., 5.5, 7)
t2 = np.linspace(0., 5.5, 11)
x0 = np.eye(3)
Y1, g = mlfm.sim(x0[0, :], t1, beta=beta)
Y2, _ = mlfm.sim(x0[1, :], t2, beta=beta, latent_forces=g)

mlfm._setup_times([t1, t2], h=.25, multi_output=True)

#################################################################
#
# We now consider the iterative process for constructing an
# estimate of the latent force

def g_em_fit(g, beta, ifx, mlfmsa):
    P = mlfmsa._K(g, beta, ifix)

    # get the data

# data preprocessing

mlfm.X_train_ = [t1, t2]
mlfm.Y_train_ = [Y1, Y2]

#mu_ivp = mlfm.mu_ivp_init([0, 5, 9])
