import scipy.stats
import numpy as np

class ProbabilityDistribution:

    def __init__(self, pdf, rvs, dim=1):
        self.dim = dim
        self._pdf = pdf
        self._rvs = rvs

    def pdf(self, x, **kwargs):
        return self._pdf(x, **kwargs)


    def __mul__(self, other):
        if isinstance(other, ProbabilityDistribution):
            return ProductProbabilityDistribution(self, other)

        else:
            raise TypeError


class ProductProbabilityDistribution:

    def __init__(self, *probs):
        self._probdists = probs


    def pdf(self, xx):
        try:
            ps = [p.pdf(x) for x, p in zip(xx, self._probdists)]

        except:
            pass


# inside GP
# GP.mhupdate():
#   updates the kernel hyper parameter using mh mcmc
# GP.hmcupdate()
class Proposal(ProbabilityDistribution):

    def __init__(self, pdf, rvs, dim=1):
        super(Proposal, self).__init__(pdf, rvs, dim=dim)

    def rvs(self, xcur):
        return self._rvs(xcur)

    def pdf(self, xnew, xcur):
        return self._pdf(xnew, xcur)

    @classmethod
    def normalrw(cls, scale):
        if isinstance(scale, float):
            dim = 1
        else:
            dim = len(scale)

        cls.scale = scale

        return cls(lambda x, xcur: scipy.stats.norm.pdf(x, loc=xcur, scale=scale),
                   lambda xcur: xcur + np.random.normal(scale=scale),
                   dim=dim)

q = Proposal.normalrw(0.5)

#xcur = 0.5
#xnew = q.rvs(xcur)
# change of idea - output dimension should be one 

class GP:
    def __init__(self, kernel,
                 hyperpar_prior=None,                 
                 hyperpar_proposal=None):
        
        self.hyperpar_prior = hyperpar_prior
        self.hyperpar_proposal = hyperpar_proposal

    def mh_update(self):
        """Update the parameter using MCMC methods
        """
        pass
