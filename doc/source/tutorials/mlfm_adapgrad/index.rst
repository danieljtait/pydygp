
MLFM Adaptive Gradient Matching
===============================

Bayesian adaptive gradient matching methods are a class of approximate
inference methods developed to handle the very general class of,
possibly nonlinear, ordinary differential equations

.. math::

   \dot{\mathbf{x}}(t) = f(x ; \boldsymbol{\theta}),

They work by placing independent Gaussian process priors on each dimension
of the state variable, and applying a product of experts assumption.
Leads to the joint likelihood term

.. math::

   \begin{align}
   p(\mathbf{X} \mid \mathbf{g} ) \propto
   \prod_{k=1}^{K} \exp\bigg\{
   &-\frac{1}{2}
   (\mathbf{f}_k - \mathbf{m}_{\dot{x}_k|x_k})^{\top}
   (\mathbf{C}_{\dot{x}_k|\dot{x}} + \gamma I)^{-1}
   (\mathbf{f}_k - \mathbf{m}_{\dot{x}_k|x_k}) \\
   &-\frac{1}{2}\mathbf{x}_k^{\top}\mathbf{C}_{x_k}\mathbf{x}
   \bigg\},
   \end{align}

where :math:`\mathbf{f}_k` is the vector of components of the evolution equation,
the entries of which are given by

.. math::

   f_{kn} = 
   \sum_{r=0}^{R} g_{rn} \sum_{d=1}^D \beta_{rd}
   \sum_{j=1}^{K} L_{dkj} x_{jn},

One interesting point from the perspective of identifiability of models of these
types is that the likelihood-term will remain invariant under choices of
:math:`\mathbf{g}, \boldsymbol{\beta}` such that :math:`\mathbf{f}_k` remains
invariant.

.. toctree::
   :maxdepth: 1

   ../mlfm_adapgrad_tutorials/plot_mlfmag
   ../mlfm_adapgrad_tutorials/plot_mlfmagvar

All of the adative gradient matching methods proceed from this conditional density

.. include:: ../../gen_modules/backreferences/pydygp.linlatentforcemodels.MLFMAdapGrad.examples
.. raw:: html

    <div style='clear:both'></div>
