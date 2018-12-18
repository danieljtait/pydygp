"""

Product Topology of MLFM Models
===============================

This note demonstrates the use of the :py:obj:`*` operator
to construct the Cartesian product of MLFM models. The
idea is to combine MLFM with distinct topologies, but a
common set of latent forces.

.. note::

   Still in development -- working out the most natural way of calling fit on the
   product object. So far all of this note does it demonstrate that the product
   operator now gathers the MLFM objects up nicely and provides a nice method
   for flattening them back out.

"""
import numpy as np
import matplotlib.pyplot as plt
from pydygp.liealgebras import so
from pydygp.linlatentforcemodels import MLFMAdapGrad
from pydygp.probabilitydistributions import (Normal, Laplace)
from sklearn.gaussian_process.kernels import RBF

N_manifolds = 3

mlfm1 = MLFMAdapGrad(so(2), R=1, lf_kernels=[RBF(), ])
mlfm2 = MLFMAdapGrad(so(2), R=1, lf_kernels=[RBF(), ])
mlfm3 = MLFMAdapGrad(so(2), R=1, lf_kernels=[RBF(), ])

mmlfm = mlfm1 * mlfm2 * mlfm3
mlfms = mmlfm.flatten()
for item in mlfms:
    print(hasattr(item, 'fit'), item.dim.D)

######################################################################
#
# Simulation
# ==========
#
# We simulate from this model by specifying the values of the
# :py:obj:`beta_i` for each of the manifolds defining the product.
# These will then typically be passed to simulate in the form of a
# list, tuple. There is typically no reason to expect each of the
# :py:obj:`beta_i` to be the same shape for the distinct models.

x0_0 = [1., 0.]
x0_1 = [0., 1.]
x0_2 = [1., 0.]
beta_0 = np.array([[0., ], [1, ]])
beta_1 = np.array([[-0.3, ], [1, ]])
beta_2 = np.array([[0., ], [2, ]])

beta = (beta_0, beta_1, beta_2)

tt = np.linspace(0., 5., 15)
Y, lf = mmlfm.sim((x0_0, x0_1, x0_2), tt, beta=beta)

######################################################################
#
# Model Fitting
# =============
#
fitopts = {'loggamma_is_fixed': True}
fitopts['logpsi_is_fixed'] = True
fitopts['beta_is_fixed'] = False
fitopts['logphi_is_fixed'] = True
fitopts['logtau_is_fixed'] = True
for i in range(N_manifolds):
    fitopts['beta_{}_init'.format(i)] = beta[i] 
    #fitopts['beta_{}_prior'.format(i)] = Laplace() * 2 #Normal(scale=1.)*2

ytrain = [Yq.T.ravel() for Yq in Y]
mmlfm.fit(tt, ytrain, **fitopts)

fig, ax = plt.subplots()
ttd = np.linspace(tt[0], tt[-1], 100)
ax.plot(tt, mlfm1.mapres.g, '+')
ax.plot(ttd, lf[0](ttd), 'k-', alpha=0.4)

# check the reconstruction
from scipy.interpolate import interp1d
lfhat = interp1d(tt, mlfm1.mapres.g.ravel(),
                 kind='cubic', fill_value='extrapolate')
ax.plot(ttd, lfhat(ttd), 'C0-')

fig, ax = plt.subplots()
ax.plot(tt, Y[0], 's')
ax.plot(ttd, mlfm1.sim(x0_0, ttd, beta=mlfm1.mapres.beta,
                       latent_forces=(lfhat,))[0], '-')
print(mlfm1.mapres.beta)

fig, ax = plt.subplots()
#ax.plot(tt, Y[1], 's')
#ax.plot(ttd, mlfm2.sim(x0_1, ttd, beta=mlfm2.mapres.beta,
#                       latent_forces=(lfhat,))[0], '-')
ax.plot(ttd, beta_1[0, 0] + beta_1[1, 0] * lf[0](ttd), 'k-')
ax.plot(ttd, mlfm2.mapres.beta[0, 0] + mlfm2.mapres.beta[1, 0]*lfhat(ttd), 'C0-')


fig, ax = plt.subplots()
#ax.plot(tt, Y[2], 's')
ax.plot(ttd, beta_2[0, 0] + beta_2[1, 0] * lf[0](ttd), 'k-')
ax.plot(ttd, mlfm3.mapres.beta[0, 0] + mlfm3.mapres.beta[1, 0]*lfhat(ttd), 'C0-')
#ax.plot(ttd, mlfm3.sim(x0_2, ttd, beta=mlfm3.mapres.beta,
#                       latent_forces=(lfhat,))[0], '-')
print(mlfm3.mapres.beta)
plt.show()
