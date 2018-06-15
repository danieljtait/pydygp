import numpy as np
import scipy.stats

class BaseProbabilityDistribution:
    def __init__(self, pdf=None, logpdf=None, rvs=None):
        self._pdf = pdf
        self._logpdf = logpdf
        self._rvs = rvs

    def pdf(self, x, *args, **kwargs):
        return self._pdf(x, *args, **kwargs)

    def logpdf(self, x, *args, **kwargs):
        return self._logpdf(x, *args, **kwargs)

    def rvs(self, size, *args, **kwargs):
        return self._rvs(size, *args, **kwargs)


class Gamma(BaseProbabilityDistribution):

    def __init__(self, a=1, b=1):

        self.a = a
        self.b = b
        
        def _pdf(x):
            return gamma.pdf(x, a=self.a, scale=1/self.b)

        def _logpdf(x):
            return gamma.logpdf(x, a=self.a, scale=1/self.b)

        def _rvs(size):
            return gamma.rvs(size=size, a=self.a, scale=1/self.b)
        
        super(Gamma, self).__init__(pdf=_pdf, logpdf=_logpdf, rvs=_rvs)


class RWProposal:
    def __init__(self, scale):
        self.scale = scale

    def rvs(self, xcur):
        xnew = np.random.normal(loc=xcur, scale=self.scale)
        return xnew
