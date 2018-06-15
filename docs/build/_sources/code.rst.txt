##########################
Documentation for the Code
##########################

.. automodule:: pydygp
.. automodule:: pydygp.kernels

.. automodule:: pydygp.linlatentforcemodels

=================================

.. autoclass:: pydygp.kernels.Kernel
   :members: SquareExponKernel

=================================
The probability density function is given by

.. math::
   p(x) = \prod_{k=1}^{K}\exp\left\{-\frac{1}{2} \mathbf{x}^T\mathbf{C}_{xx}^{-1}\mathbf{x} - \frac{1}{2}(\mathbf{f}_k-\mathbf{M}_k\mathbf{x}_k)\mathbf{S}_k^{-1}(\mathbf{f_k}-\mathbf{M}_k\mathbf{x}_k) \right\}
   :label: xpostprob

.. autoclass:: pydygp.linlatentforcemodels.MLFM_MH_AdapGrad
	       
Utility functions

Utility function which gets called to return the matrices :math:`\mathbf{A}, \mathbf{B}, \mathbf{C}` in equation :eq:`xpostprob`


.. autofunction:: pydygp.linlatentforcemodels.mlfm_mh_adapgrad.gpdx

.. autofunction:: pydygp.linlatentforcemodels.mlfm_mh_adapgrad.dx_gp_condmats
