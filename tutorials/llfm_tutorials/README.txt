
.. _tutorials-index:

Linear Latent Force Models
==========================
This module provides methods to carry out simulation and fitting of latent force models, which are broadly taken to be time dependent linear ODEs driven by a set of smooth Gaussian processes.

.. _tutorials-mlfm:

Multiplicative Latent Force Model
---------------------------------

Multiplicative latent force models are time dependent linear ODEs of the form

.. math::

   \dot{X}(t) = A(t)X(t), \qquad A(t) = A_0 + \sum_{r=1}^R g_r(t) A_r,

where :math:`\{ g_r(t) \}_{r=1}^R` are a set of independent smooth scalar Gaussian processes, and :math:`\{A_r \}_{r=0}^{R}` are a set of square :math:`K\times K`. The following examples demonstrate the process of constructing these models and demonstrates the structure preserving properties of this model as well as how to carry out inference.

MLFM AdapGrad
-------------

.. _tutorials-index-mlfmag-par:

Parameters
~~~~~~~~~~
The model parameters we are most interested in include the
