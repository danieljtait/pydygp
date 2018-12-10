"""

Approximate Density
===================

When using the method of successive approximation we construct
a regression model 

.. math::

   p(\\mathbf{x} \\mid \\mathbf{g}, \\boldsymbol{\\beta})
   = \\mathcal{N}(
   \\mathbf{x}
   \\mid \\mathbf{P}^{M} \\boldsymbol{\\mu}_0,
   \\alpha \\mathbf{I})
   
The idea is to construct an approximation to this density by
introduction each of the successive approximations

.. math::

   \\mathbf{z}_{i} = \\mathbf{P}\\mathbf{z}_{i-1},

the idea being that knowing the complete set of approximations
:math:`\{ z_0,\ldots,z_M\}` we can solve for the latent variables
by rearranging the linear equations, instead of manipulating the
polynomial mean function.

For this conversion to work we need to introduce a regularisation
parameter :math:`\lambda > 0` and then define :math:`\mathcal{N}(
\\mathbf{z}_{i} \\mid \\mathbf{P}\\mathbf{z}_{i-1}, \\lambda
\\mathbf{I})`, once we do this we can write log-likelihood of the
state variables

.. math::

   \\log = -\\frac{\lambda}{2} \\sum_{i=1}^{M}
   \\left(
   \\mathbf{z}_{i-1}^{\\top}\\mathbf{P}^{\\top}\\mathbf{P}\\mathbf{z}_{i-1}
   - 2\\mathbf{z}_{i}^{\\top}\\mathbf{P}\\mathbf{z}_{i-1}
   \\right)

Now the matrices :math:`\\mathbf{P}` are linear in the parameters
which means that after vectorisation they can be represented as

.. math::

   \\operatorname{vec}(\\mathbf{P}) = \\mathbf{V}\\mathbf{g} + \\mathbf{v}_0

.. note::

   The matrices :math:`\\mathbf{P}` and their affine representations are
   most easily written compactly using kronecker products, unfortunately
   these are not necessarily the best computational representations and
   there is a lot here that needs refining.

Linear Gaussian Model
---------------------
We take a break in the model to now discuss how to start putting some
of the ideas discussed above into code. For the Kalman Filter we are
going to use the code in the
`PyKalman package <https://pykalman.github.io/>`_, but hacked a little
bit to allow for filtering and smoothing of independent sequences
with a common transition matrix.
"""
import numpy as np
from pydygp.linlatentforcemodels import MLFMSA

def _get_kf_statistics(X, kf):
    """ Gets
    """
    # the mean, cov and kalman gain matrix
    means, cov, kalman_gains = kf.smooth(X)
    # pairwise cov between Cov{ z[i], z[i-1]
    # note pairwise_covs[0] = 0  - it gets ignored
    pairwise_covs = kf._smooth_pair(covs, gains)
    

######################################
# Expectation Maximisation
# ------------------------
#
# So we have introduced a large collection of unintersting latent variables,
# the set of successive approximations :math:`\{ z_0, \ldots, z_M \}`, and
# so we need to integrate them out. If we define the statistics
#
# .. math::
#
#    \boldsymbol{\Psi}_0 = \sum_{i=1}^{M} \langle \mathbf{z}_{i-1}
#    \mathbf{z}_{i-1}^{\top} \rangle_{q(Z)}, \quad
#    \boldsymbol{\Psi}_1 = \sum_{i=1}^{M} \langle \mathbf{z}_{i}
#    \mathbf{z}_{i-1}^{\top} \rangle_{q(Z)}
#
# Then the objective function of the `M-step` becomes
#
# .. math::
#
#    Q(\mathbf{g}, \mathbf{g}^{old}) =
#    -\frac{1}{2} \mathbf{g}^{\top}
#    \left( \mathbf{V}^{\top}
#    (\boldsymbol{\Psi}_0 \otimes \mathbf{I}_{NK})\mathbf{V} +
#    \lambda^{-1} \mathbf{C}^{-1} \right)\mathbf{g} - 2

