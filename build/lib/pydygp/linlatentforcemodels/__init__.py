"""
==========================
Linear Latent Force Models
==========================

.. module:: pydygp.linlatentforcemodels

Linear ordinary differential equations driven by smooth Gaussian processes.


Multiplicative Forces
=====================

.. autosummary::
   :toctree:

   BaseMLFM     -- Base class for the MLFM.
   MLFMAdapGrad -- MLFM using adaptive gradient matching.
   GibbsMLFMAdapGrad -- Gibbs sampling for MLFM using adaptive gradient matching.

"""
from .mlfm import BaseMLFM
from .mlfmadapgrad import MLFMAdapGrad, GibbsMLFMAdapGrad, VarMLFMAdapGrad
