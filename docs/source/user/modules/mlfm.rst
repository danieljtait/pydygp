.. _mlfm

==================================
Multiplicative Latent Force Models
==================================

.. currentmodule:: pydygp.linlatentforcemodels

Simulating the MLFM
===================

We consider an example on the sphere and so we first create the infinitesimal rotation matrices

   >>> import numpy as np
   >>> from pydygp.gaussianprocesses import GaussianProcess
   >>> from pydygp.linlatentforcemodels import MLFM
   >>> # make the infinitesimal rotation matrices
   >>> Lx = np.array([[0., 0., 0.], [0., 0.,-1.], [0., 1., 0.]])
   >>> Ly = np.array([[0., 0.,-1.], [0., 0., 0.], [1., 0., 0.]])   
   >>> Lz = np.array([[0.,-1., 0.], [1., 0., 0.], [0., 0., 0.]])

Lets consider a simple model involing a constant rotation around the north pole of the shere along with a random fluctation

   .. math::

      \dot{\mathbf{x}}(t) = \left[\mathbf{L}_z + g(t)\cdot(\mathbf{L}_x + \mathbf{L}_y)\right]\mathbf{x}(t)

We then make the model by passing a list of the structure matrices to MLFM

   >>> gp = GaussianProcess("sqexp", kpar=[0.1, 1.])
   >>> mlfm = MLFM([Lz, Lx+Ly])

The results are displayed
   
.. plot::

   >>> import numpy as np
   >>> import matplotlib.pyplot as plt
   >>> from pydygp.gaussianprocesses import GaussianProcess
   >>> from pydygp.linlatentforcemodels import MLFM
   >>> # make the infinitesimal rotation matrices
   >>> np.random.seed(125)
   >>> Lx = np.array([[0., 0., 0.], [0., 0.,-1.], [0., 1., 0.]])
   >>> Ly = np.array([[0., 0.,-1.], [0., 0., 0.], [1., 0., 0.]])   
   >>> Lz = np.array([[0.,-1., 0.], [1., 0., 0.], [0., 0., 0.]])
   >>> gp = GaussianProcess("sqexp", kpar=[.1, 1.])
   >>> gp.jitter = False
   >>> mlfm = MLFM([Lz, Lx+Ly])
   >>> tt = np.linspace(0., 5., 10)
   >>> X, gval, ttd, xxd = mlfm.sim([0., 0., 1.], tt, gps=(gp, ), return_gp=True)
   >>> fig = plt.figure()
   >>> ax1 = fig.add_subplot(121)
   >>> ax1.plot(tt, X)
   >>> ax2 = fig.add_subplot(122)
   >>> ax2.plot(ttd, gval, '+')
   >>> plt.show()
