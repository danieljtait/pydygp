"""
Variational Inference
=====================

This example presents an illustration of using the MLFM to
learn the model

.. math::

    \dot{\mathbf{x}}(t) = \mathbf{A}(t)\mathbf{x}(t)

where :math:`A(t) \in \mathfrak{so}(3)` and :math:`\| x_0 \| = 1`.

This note will also demonstrate the process of holding certain variables
fixed as well as defining priors


"""
import matplotlib.pyplot as plt
import numpy as np
from pydygp.probabilitydistributions import (GeneralisedInverseGaussian,
                                             InverseGamma,
                                             Normal)
from sklearn.gaussian_process.kernels import RBF
from pydygp.liealgebras import so
from pydygp.linlatentforcemodels import (MLFMAdapGrad,
                                         GibbsMLFMAdapGrad,
                                         VarMLFMAdapGrad)
np.random.seed(15)
np.set_printoptions(precision=3, suppress=True)
###########################################################################
# Our first step is to initialise the models and then simulate some data.
#
# Make the model
vmlfm = VarMLFMAdapGrad(so(3), R=1, lf_kernels=[RBF(),])
gmlfm = GibbsMLFMAdapGrad(so(3), R=1, lf_kernels=[RBF(),])

beta = np.row_stack(([0.]*3,
                     np.random.normal(size=3)))

# simulate some initial conditions
x0 = np.random.normal(size=6).reshape(2, 3)
x0 /= np.linalg.norm(x0, axis=1)[:, None]

# Time points to solve the model at
tt = np.linspace(0., 6, 7)

# Data and true forces
Data, g0 = vmlfm.sim(x0, tt, beta=beta, size=2)
# vectorised and stack the data
Y = np.column_stack((y.T.ravel() for y in Data))

###########################################################################
# Specifying priors
# -----------------
# .. currentmodule:: pydygp.probabilitydistributions
#
# The prior should have a loglikelihood(x, eval_gradient=False) method
# which returns the loglikelihood of the prior variable at x and
# and optionally its gradient.
#
# Preexisting priors are contained in :py:mod:`pydygp.probabilitydistributions`
#
# for example there is the class :ref:`ProbabilityDistribution`
logpsi_prior = GeneralisedInverseGaussian(a=5, b=5, p=-1).logtransform()
loggamma_prior = [InverseGamma(a=0.001, b=0.001).logtransform(),]*vmlfm.dim.K

beta_prior = Normal(scale=1.) * beta.size

fitopts = {'logpsi_is_fixed': True, 'logpsi_prior': logpsi_prior,
           'loggamma_is_fixed': False, 'loggamma_prior': loggamma_prior,
           'beta_is_fixed': True, 'beta_prior': beta_prior,
           'beta0': beta,
           }

# Fit the model
res, Eg, Covg = vmlfm.varfit(tt, Y, **fitopts)

Grv = gmlfm.gibbsfit(tt, Y, **fitopts, mapres=res)

Lapcov = res.optimres.hess_inv[:vmlfm.dim.N*vmlfm.dim.R,
                               :vmlfm.dim.N*vmlfm.dim.R]

fig, ax = plt.subplots()
#ax.plot(tt, res.g.T, '+')
ax.plot(tt, Grv['g'].T, 'k+', alpha=0.2)
#ax.plot(tt, Eg, 'o')
#ax.errorbar(tt, res.g.T, yerr = 2*np.sqrt(np.diag(Lapcov)), fmt='s')
#ax.errorbar(tt, Eg, yerr = 2*np.sqrt(np.diag(Covg[..., 0, 0])), fmt='o')

ttdense = np.linspace(0., tt[-1])
ax.plot(ttdense, g0[0](ttdense), 'k-', alpha=0.2)
fpred, fstd = vmlfm.predict_lf(ttdense, return_std=True)

ax.plot(ttdense, fpred[0, :], '-.')
ax.fill_between(ttdense,
                fpred[0, :] + 2*fstd[0, :],
                fpred[0, :] - 2*fstd[0, :],
                alpha=0.3)

plt.show()



"""
ttdense = np.linspace(tt[0], tt[-1], 50)
Cff_ = vmlfm.latentforces[0].kernel_(ttdense[:, None], tt[:, None])
Cf_f_ = vmlfm.latentforces[0].kernel_(tt[:, None])
Cf_f_[np.diag_indices_from(Cf_f_)] += 1e-5
Lf_f_ = np.linalg.cholesky(Cf_f_)

from scipy.linalg import cho_solve
gpred = Cff_.dot(cho_solve((Lf_f_, True), Eg))
print(np.sqrt(np.diag(Covg[..., 0, 0])))
ax.plot(ttdense, gpred, 'r-.')
"""
