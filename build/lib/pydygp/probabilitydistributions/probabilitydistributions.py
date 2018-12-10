import numpy as np
from scipy.special import gammaln

class ProbabilityDistribution:

    def logpdf(self, x, eval_gradient=False):
        raise NotImplementedError

    def __mul__(self, b):
        if isinstance(b, ProbabilityDistribution) \
           or isinstance(b, ProductProbabilityDistribution):
            return ProductProbabilityDistribution(self, b)

    @property
    def output_dim(self):
        return 1

class ProductProbabilityDistribution:

    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def logpdf(self, x, eval_gradient=False):
        x = np.asarray(x)
        x1 = x[:self.p1.output_dim]
        x2 = x[self.p1.output_dim:]
        if eval_gradient:
            lp1, lp1_dx = self.p1.logpdf(x1, True)
            lp2, lp2_dx = self.p2.logpdf(x2, True)

            if isinstance(lp1_dx, float):
                lp1_dx = [lp1_dx]
            if isinstance(lp2_dx, float):
                lp2_dx = [lp2_dx]

            return lp1 + lp2, np.concatenate((lp1_dx, lp2_dx))

        else:
            lp1 = self.p1.logpdf(x1)
            lp2 = self.p2.logpdf(x2)
            return lp1 + lp2

    @property
    def output_dim(self):
        return self.p1.output_dim + self.p2.output_dim

    def __mul__(self, b):
        if isinstance(b, ProbabilityDistribution) or \
           isinstance(b, ProductProbabilityDistribution):
            return ProductProbabilityDistribution(self, b)

class ExpInvGamma(ProbabilityDistribution):

    def __init__(self, a=1, b=1):
        self.a=a
        self.b=b

    def logpdf(self, x, eval_gradient=False):
        ex = np.exp(x)

        lpdf = -self.a*x - self.b/ex
        lpdf += self.a*np.log(self.b) - gammaln(self.a)
        
        if eval_gradient:
            lpdf_dx = -self.a + self.b/ex
            return lpdf, lpdf_dx

        return lpdf

class ExpGeneralisedInvGaussian(ProbabilityDistribution):

    def __init__(self, a=1, b=1, p=-1):
        self.a = a
        self.b = b
        self.p = p

    def logpdf(self, x, eval_gradient=False):
        ex = np.exp(x)

        lpdf = self.p*x - .5*(self.a*ex + self.b/ex)

        if eval_gradient:
            lpdf_dx = self.p - .5*(self.a*ex - self.b/ex)
            return lpdf, lpdf_dx

        return lpdf
