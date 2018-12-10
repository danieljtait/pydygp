
.. _tutorials-index:

Multiplicative Latent Force Models
==================================
This module provides methods to carry out simulation and fitting of
latent force models, which are broadly taken to be time dependent
linear ODEs driven by a set of smooth Gaussian processes which are
allowed to interact multiplicatively with the state
variable, and so the name Multiplicative Latent Force Models (MLFM)
to differ them from the case with an additive forcing term
which are discussed in :ref:`lfm-tutorials-index`

.. _tutorials-mlfm:

Multiplicative Latent Force Model
---------------------------------

Multiplicative latent force models are time dependent linear ODEs of the form

.. math::

   \dot{X}(t) = A(t)X(t), \qquad A(t) = A_0 + \sum_{r=1}^R g_r(t) A_r,

where :math:`\{ g_r(t) \}_{r=1}^R` are a set of independent smooth scalar Gaussian processes, and :math:`\{A_r \}_{r=0}^{R}` are a set of square :math:`K\times K` matrices. Furthermore it may also be the case that for each of the structure matrices :math:`A_r` we have

.. math::

   A_r = \sum_{d} \beta_{rd} L_d,

for some common set of shared basis matrices :math:`\{ L_d \}_{d=1}^{D}` -- typically these will be chosen to form a basis of some Lie algebra.

The following tutorials demonstrate the process of constructing these models as well as demonstrating the possible structure preserving properties of this model as well as how to carry out inference.

MLFM AdapGrad
-------------

Joint Density
~~~~~~~~~~~~~

For the MLFM AdapGrad model the likelihood of a set of state variables may be written

.. math::

   p(\mathbf{X} \mid \mathbf{g} ) \propto \prod_{k=1}^{K} \exp\left\{-\frac{1}{2}(f_k - m_{\dot{x}_k|x_k})^{\top}(C_{\dot{x}_k|\dot{x}} + \gamma I)^{-1}(f_k - m_{\dot{x}_k|x_k})\right\}

.. _tutorials-index-mlfmag-par:

Parameters
~~~~~~~~~~
The model parameters we are most interested in include the

+-----------------------------+-----------------------------------------+
| :math:`\mathbf{g}`          | The (vectorised) latent force variable  |
+-----------------------------+-----------------------------------------+
| :math:`\boldsymbol{\psi}`   | Hyperparameters of the latent force GPs |
+-----------------------------+-----------------------------------------+
| :math:`\boldsymbol{\phi}`   | Hyperparameters of the latent state GPs |
+-----------------------------+-----------------------------------------+
| :math:`\boldsymbol{\gamma}` | ODE model `temperature`       parameter |
+-----------------------------+-----------------------------------------+
