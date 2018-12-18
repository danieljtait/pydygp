"""
===============================================================
Linear Latent Force Models (:mod:`pydygp.linlatentforcemodels`)
===============================================================

.. currentmodule:: pydygp.linlatentforcemodels

Linear ordinary differential equations driven by smooth Gaussian processes.


Multiplicative Forces
=====================

.. autosummary::
   :toctree: _autosummary

   BaseMLFM     -- Base class for the MLFM.
   MLFMAdapGrad -- MLFM using adaptive gradient matching.
   GibbsMLFMAdapGrad -- Gibbs sampling for MLFM using adaptive gradient matching.
   MLFMSuccApprox -- MLFM using successive approximations.

"""
from .mlfm import BaseMLFM, Dimensions
from .lfmorder2 import LFMorder2, LFMorder2Kernel
from .mlfmadapgrad import MLFMAdapGrad, GibbsMLFMAdapGrad, VarMLFMAdapGrad
from .mlfmsuccapprox import MLFMSuccApprox, VarMLFMSuccApprox
from .mlfmsamoe import MLFMSA
from .mlfmsamoe2 import MLFMSAMix
from .mlfmmixsa import MLFMMixSA
from .kalmanfilter import KalmanFilter
