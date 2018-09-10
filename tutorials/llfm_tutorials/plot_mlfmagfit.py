"""
.. _tutorials-mlfm-fit:

MLFM-AG Model Fitting
=====================

.. currentmodule:: pydygp.linlatentforcemodels

This note decribes hot to carry out the porcess of carrying out MAP parameter
estimation for the MLFM model using the Adaptive Gradient matching approximation :class:`MLFMAdapGrad` 
"""
import numpy as np
import matplotlib.pyplot as plt
from pydygp.linlatentforcemodels import MLFMAdapGrad

##############################################################################
#
# Model Setup
# ~~~~~~~~~~~
# We are going to demonstrate using the example we have already described in
# :ref:`tutorials-mlfm-sim` and the steps described there for setting up
# this model and simulating some data. The model was a simple ODE on the unit
# circle
#
# .. math::
#
#    S^1 = \{ x \in \mathbb{R}^2 \; : \; \| x \| = 1 \},
#
# We can write a linear ode on :math:`S^1` as
#
# .. math::
#
#    \begin{bmatrix} \dot{x}(t) \\ \dot{y}(t) \end{bmatrix}
#    = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}
#      \begin{bmatrix} x(t) \\ y(t) \end{bmatrix}.
#
# We can set up this model by and simulate some observations by
#
import pydygp.liealgebras

np.random.seed(7)  # seed for reproducability

so2 = pydygp.liealgebras.so(2)  # import the Lie algebra
mlfm = MLFMAdapGrad(so2, R=1)   # Build the model with 0 offset matrix

tt = np.linspace(0., 5., 11)    # time data points
Y, gtrue = mlfm.sim([1., 0], tt)    # Simulation result

##############################################################################
# Model Fit
# ~~~~~~~~~
# The API is designed to be as simple as possible and so for reasonably simple
# models we should get reasonable results by simply calling

res0 = mlfm.fit(tt, Y)
##############################################################################
# this returns a simple :class:`namedtuple` object which has attributes with
# names corresponding to the parameters described in
# :ref:`tutorials-index-mlfmag-par`. We can compare the point estimates with
# the true latent force functions
ttdense = np.linspace(tt[0], tt[-1], 21)  # make a set of dense times
fig0, ax = plt.subplots()
ax.plot(ttdense, gtrue[0](ttdense), 'k-', alpha=0.5)
ax.plot(tt, res0.g, '+')

##############################################################################
# Sample Density Dependenance
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The adaptive gradient matching processes depend on the use of GP
# interpolators of the unknown state trajectories, and therefore
# we might express the accuracy of these methods to decreae not with the
# sample size but with the space between samples.
# 
# Dense Predictions
# ~~~~~~~~~~~~~~~~~
mlfm2 = MLFMAdapGrad(so2, R=1)
data_inds = np.linspace(0, ttdense.size-1, tt.size, dtype=np.intp)
mlfm2._setup_times(tt, tt_aug=ttdense, data_inds=data_inds)
res_dense = mlfm2.fit(tt, Y, logpsi_is_fixed=True)

fig, ax = plt.subplots()
ss = np.linspace(tt[0], tt[-1])
ax.plot(ss, gtrue[0](ss), 'k-', alpha=0.2)
ax.plot(mlfm.ttc, res0.g, '+')
ax.plot(mlfm2.ttc, res_dense.g, 'o')

plt.show()
