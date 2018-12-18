"""
==================================================================
Probability Distributions (:mod:`pydygp.probabilitydistributions`)
==================================================================

.. currentmodule:: pydygp.probabilitydistributions

Univariate Distributions
------------------------
Univariates distribution may be multiplied to produce a multivariate
distribution where each component is independent

>>> from pydygp.probabilitydistributions import Normal
>>> from scipy.stats import norm
>>> p = Normal()
>>> ppp = p * 3
>>> ppp.logpdf([0., 1., 2.])
>>> norm.logpdf([0., 1., 2.])

.. autosummary::
   :toctree: generated/
   :template: class.rst

   UnivariateProbabilityDistribution
   Normal
   Laplace
   ChiSquare
   Gamma
   InverseGamma
   GeneralisedInverseGaussian

Multivariate Distributions
--------------------------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   MultivariateNormal

"""
from .probabilitydistributions import (ProbabilityDistribution,
                                       UnivariateProbabilityDistribution,
                                       Gamma,
                                       InverseGamma,
                                       ChiSquare,
                                       Normal,
                                       Laplace,
                                       MultivariateNormal,
                                       GeneralisedInverseGaussian,
                                       ExpGamma,
                                       ExpInvGamma,
                                       ExpGeneralisedInvGaussian)
