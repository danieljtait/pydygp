"""

.. _tutorials-mlfm-fit-wpriors:

Specifying Priors
=================

.. currentmodule:: pydygp.linlatentforcemodels

This note provides more detail into the process of carrying out model fitting using
the MLFM model with the adaptive gradient matching process.

As usual we carry out the initial imports and construct some data, one interesting
feature of this example is that we are going

"""
import numpy as np
import matplotlib.pyplot as plt
from pydygp.linlatentforcemodels import MLFMAdapGrad
import pydygp.liealgebras
from sklearn.gaussian_process.kernels import RBF
np.random.seed(7)
so3 = pydygp.liealgebras.so(3)
lf_kernels = [RBF(), ]
beta = np.random.normal(size=6).reshape(2, 3)

###############################################################################
# One interesting feature of this example is that we are going to simulate the
# model at a collection of different initial values with a common latent forces.
# We chose our initial conditions to be uniformly distribution on te surface of
# the sphere. 
nsim = 3
x0 = np.random.normal(size=3*3).reshape(nsim, 3)
x0 /= np.linalg.norm(x0, axis=1)[:, None]

mlfm = MLFMAdapGrad(so3, R=1,
                    lf_kernels=lf_kernels,
                    is_beta_fixed=False)

# Time points to simulate on
tt = np.linspace(0., 5., 15)
Data, gtrue = mlfm.sim(x0, tt, beta, size=x0.shape[0])

####################################################################
#
# Holding Parameters Fixed
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# The naming convention used for accessing parameters of interest is
# typically of the form ...
#
# For example to fit the model holding `beta' fixed so we combine the above to
# produce our model and then carry out the fit treating the noise parameters,
# and the :math:`$\log(\gamma)$` parameters fixed.

# vectorise the data
Y = np.column_stack((data.T.ravel() for data in Data))

fig, ax = plt.subplots()
ax.plot(tt, Data[0], 'k-', alpha=0.5)
# corrupt Y with some iid noise
Y += np.random.normal(size=Y.size, scale=0.1).reshape(Y.shape)
ax.plot(tt, Y[:, 0].reshape((3, tt.size)).T, '+')

mapres = mlfm.fit(tt, Y,
                  logtau_is_fixed=False,
                  beta0 = beta,
                  beta_is_fixed=True)

from pydygp.probabilitydistributions import ChiSquare
logtau_prior = ChiSquare(df=1).scaletransform(0.1**2)  # X/C ~ N(0., 0.1**2)
logtau_prior = logtau_prior.logtransform()
logtau_prior = logtau_prior.scaletransform(-1)

q = ChiSquare(df=1).scaletransform(0.05**2)
q = q.logtransform()
q = q.scaletransform(-1)

xx = np.linspace(3, 15, 100)
fig, ax = plt.subplots()
ax.plot(xx, np.exp([logtau_prior.logpdf(x) for x in xx]), 'b--')
ax.plot(xx, np.exp([q.logpdf(x) for x in xx]), 'r--')

logtau_prior = logtau_prior * 3

mapres2 = mlfm.fit(tt, Y,
                   logtau_is_fixed=False,
                   logtau_prior=logtau_prior,
                   beta0 = beta,
                   beta_is_fixed=True)
print(mapres.logtau)
print(mapres2.logtau)

fig, ax = plt.subplots()
ax.plot(tt, gtrue[0](tt))
ax.plot(tt, mapres.g.ravel(), 's')
ax.plot(tt, mapres2.g.ravel(), 'o')
plt.show()


