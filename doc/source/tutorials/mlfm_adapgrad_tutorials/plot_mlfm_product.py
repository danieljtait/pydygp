"""

Product Topology of MLFM Models
===============================

This note demonstrates the use of the :py:obj:`*` operator
to construct the Cartesian product of MLFM models. The
idea is to combine MLFM with distinct topologies, but a
common set of latent forces.

.. note::

   Still in development -- working out the most natural way of calling fit on the
   product object. So far all of this note does it demonstrate that the product
   operator now gathers the MLFM objects up nicely and provides a nice method
   for flattening them back out.

"""
from pydygp.liealgebras import so
from pydygp.linlatentforcemodels import MLFMAdapGrad
from sklearn.gaussian_process.kernels import RBF

mlfm1 = MLFMAdapGrad(so(2), R=1, lf_kernels=[RBF(), ])
mlfm2 = MLFMAdapGrad(so(3), R=1, lf_kernels=[RBF(), ])
mlfm3 = MLFMAdapGrad(so(4), R=1, lf_kernels=[RBF(), ])

mmlfm = mlfm1 * mlfm2
mmlfm = mmlfm * mlfm3

mlfms = mmlfm.flatten()
for item in mlfms:
    print(hasattr(item, 'fit'), item.dim.D)


fitopts = {'loggamma_is_fixed': True}

is_fixed_vars = mmlfm._fit_kwarg_parser(len(mlfms), **fitopts)
print(is_fixed_vars)

