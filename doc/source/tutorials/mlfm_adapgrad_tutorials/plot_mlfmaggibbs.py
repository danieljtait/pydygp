"""
Gibbs Sampling
==============

This example presents an illustration of the MLFM to learn the model

.. math::

   \dot{\mathbf{x}}(t)    

We do the usual imports and generate some simulated data
"""
import numpy as np
import matplotlib.pyplot as plt
from pydygp.probabilitydistributions import (Normal,
                                             GeneralisedInverseGaussian,
                                             InverseGamma)
from sklearn.gaussian_process.kernels import RBF
from pydygp.liealgebras import so
from pydygp.linlatentforcemodels import GibbsMLFMAdapGrad

np.random.seed(15)


gmlfm = GibbsMLFMAdapGrad(so(3), R=1, lf_kernels=(RBF(), ))

beta = np.row_stack(([0.]*3,
                     np.random.normal(size=3)))

x0 = np.eye(3)

# Time points to solve the model at
tt = np.linspace(0., 6., 7)

# Data and true forces
Data, lf = gmlfm.sim(x0, tt, beta=beta, size=3)

# vectorise and stack the data
Y = np.column_stack((y.T.ravel() for y in Data))

logpsi_prior = GeneralisedInverseGaussian(a=5, b=5, p=-1).logtransform()
loggamma_prior = InverseGamma(a=0.001, b=0.001).logtransform() * gmlfm.dim.K
beta_prior = Normal(scale=1.) * beta.size

fitopts = {'logpsi_is_fixed': False, 'logpsi_prior': logpsi_prior,
           'loggamma_is_fixed': False, 'loggamma_prior': loggamma_prior,
           'beta_is_fixed': False, 'beta_prior': beta_prior
           }

#gibbsRV = gmlfm.gibbsfit(tt, Y, **fitopts)