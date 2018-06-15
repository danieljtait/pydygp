#################################
Multiplicative Latent Force Model
#################################
The muliplicative latent force model is defined by the non-autonomous ordinary differential equation model

.. math::
   \frac{\operatorname{d}}{\operatorname{d}t}\mathbf{x} = \left(\mathbf{A}_0 + \sum_{r=1}^{R} g_r(t)\mathbf{A}_r \right)\mathbf{x}(t),

where each of the :math:`R` latent forces are modelled using independent Gaussian processes.

.. math::

   g_r(t) \sim \mathcal{GP}(0, k_r(s, t))

##########################
Adaptive Gradient Matching
##########################


Metropolis-Hastings Model Setup
===============================
The following functions are used to setup the probabilistic structure of the model. The values of the latent variables will not be initalised.

Section 1.1 Worked Example
--------------------------
We assume that the :class:`pydygp.linlatentforcemodels.MLFM_MH_AdapGrad` object has been initalised. Recall from before that for each of the ... we have ... .

We also want to create a lightweight prior and proposal distribution for each of the
hyperparameters :math:`\boldsymbol{\phi_k} = (\phi_{k1}, \phi_{k2})^T`. We can do this using the :class:`collections.namedtuple` class as follows

.. code::

   from collections import namedtuple

   Proposal = namedtuple('Proposal', 'rvs')


This will create a proposal, and prior method with the absolute minimal methods needed to function, namely a :function:`.rvs` method. Our proposal should take at least two positional arguments

.. code::

   # Simple isotropic random walk proposal, we also return the proposal
   # ratio q(xcurxnew)/q(xnew|xcur) = 1
   def rw_proposal(xcur, scale):
       return np.random.normal(loc=xcur, scale=1), 1
   
   scales = [0.1, 0.15, 0.1]
   phi_proposal = [Proposal(rvs=rw_proposal(s)) for s in scale]

and we also put uniform priors on the hyper-parameters

.. code::

   from scipy.stats import uniform
   phi_prior = [uniform(loc=0, scale=5)]*3
   

Let's create a pair of kernel functions

.. code::

   from pydygp.kernels import GradientKernel

   # Initalise a collection of square exponential kernels
   x_kernels = [GradientKernel.SquareExponKernel(dim=1) for k in range(3)]

   # setup the model to have square expon. kernels for the latent forces
   mlfm.x_gp_setup(x_kernels=x_kernels)

Section 1.2 Setup Functions
---------------------------

.. automethod:: pydygp.linlatentforcemodels.MLFM_MH_AdapGrad.phi_setup
.. automethod:: pydygp.linlatentforcemodels.MLFM_MH_AdapGrad.x_gp_setup

Section 1.3 Full Example
------------------------

.. literalinclude:: example.py
		
==========================
   
##########################
Adaptive Gradient Matching
##########################
.. autoclass:: pydygp.linlatentforcemodels.MLFM_AdapGrad
   :members: x_cond_posterior

