.. _gaussian_process:

==================
Gaussian Processes
==================

.. currentmodule:: pydygp.gaussianprocesses

The :class:`GaussianProcess` implements Gaussian processes.

Simple Gaussian Process Regression
==================================

.. currentmodule:: pydygp.gaussianprocesses

The :class:`._GaussianProcess` object may be used to call the basic Gaussian process regression object :class:`.BaseGaussianProcess` with a :class:`.kernels.SquareExponentialKernel` covariance functions using

   >>> from pydygp.gaussianprocesses import _GaussianProcess
   >>> gp = _GaussianProcess('sqexp_gpr')

If we then make some data

   >>> import numpy as np
   >>> tt = np.linspace(0., 5., 6)
   >>> Y = np.sin(tt)

Then we can fit the GP by calling the :func:`.BaseGaussianProcess.fit` method for a particular set of kernel parameters (or the default)

   >>> kpar = [1., 0.5]
   >>> # fit the Gaussian process for given kernel parameters
   >>> gp.fit(tt[:, None], Y, kpar=kpar)
   >>> # new set of points to sample at
   >>> ttd = np.linspace(tt[0], tt[-1], 100)
   >>> # predict the mean and covariance of the new points conditional on observations
   >>> mpred, cpred = gp.pred(ttd[:, None], return_covar=True)

Putting that altogether we have
   
.. plot::

   >>> import matplotlib.pyplot as plt
   >>> import numpy as np
   >>> from pydygp.gaussianprocesses import _GaussianProcess
   >>> tt = np.linspace(0., 5., 6)
   >>> Y = np.sin(tt)
   >>> gp = _GaussianProcess('sqexp_gpr')
   >>> gp.fit(tt[:, None], Y)
   >>> # Make a dense sample
   >>> ttd = np.linspace(0., 5., 100)
   >>> # predict the mean and covariance
   >>> mpred, cpred = gp.pred(ttd[:, None], return_covar=True)
   >>> sd = np.sqrt(np.diag(cpred))

   >>> fig, ax = plt.subplots()
   >>> ax.fill_between(ttd, mpred + 2*sd, mpred-2*sd, alpha=.2)
   >>> ax.plot(ttd, mpred, 'k-.')
   >>> ax.plot(tt, Y, 's')
   >>> plt.show()

Hyperparameter Optimisation
===========================

.. currentmodule:: pydygp.gaussianprocesses

To perform hyperparameter optimisation we call :func:`GaussianProcess.hyperpar_optim`
