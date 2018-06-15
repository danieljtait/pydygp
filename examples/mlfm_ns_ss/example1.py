import numpy as np
import matplotlib.pyplot as plt
from pydygp.kernels import Kernel
from pydygp.gaussianprocesses import (MHBayesGaussianProcess,
                                      GaussianProcess)
from pydygp.linlatentforcemodels import MLFMFactory

"""
==============
Model creation
==============
"""
Lx = np.array([[0., 0., 0.],
               [0., 0.,-1.],
               [0., 1., 0.]])

Ly = np.array([[0., 0.,-1.],
               [0., 0., 0.],
               [1., 0., 0.]])

Lz = np.array([[0.,-1., 0.],
               [1., 0., 0.],
               [0., 0., 0.]])


mlfm = MLFMFactory.onestep([Lx, Ly, Lz])


"""
===============
Data Simulation
===============
"""
np.random.seed(29)
kse = Kernel.SquareExponKernel()
gp1 = GaussianProcess(kse)
gp2 = GaussianProcess(kse)

tt = np.linspace(0., 5., 10)
x0 = [0., 0., 1.]
Y, gvals, ttd = mlfm.sim(x0, tt, gps=[gp1, gp2], return_gp=True)

"""
===========
Model Setup
===========
"""

t_setup = {'h': .2}
op_setup = {'ifix': 6}

mlfm.setup(tt,
           t_setup=t_setup,
           op_setup=op_setup)

"""
==================
Latent Force setup
==================

Initalise a collection of Gaussian process objects
and then attach them to our model.
"""
from pydygp.probdist import Gamma, RWProposal
from pydygp.kernels import _Kernel

# create the kernels...
kernels = [_Kernel('sqexp') for r in range(mlfm.dim.R)]
# and then create the latent force GPs
gps = [MHBayesGaussianProcess(kern, None, None) for kern in kernels]

# attach the gps
mlfm.g_gp_setup(gps=gps)
