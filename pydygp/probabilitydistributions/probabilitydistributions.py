import numpy as np
from scipy.special import gammaln
from scipy.linalg import cho_solve

class ProbabilityDistribution:

    def __init__(self, output_dim=1):
        self.output_dim = output_dim

    def logpdf(self, x, eval_gradient=False):
        raise NotImplementedError

    def __mul__(self, b):
        if isinstance(b, ProbabilityDistribution) \
           or isinstance(b, ProductProbabilityDistribution):
            return ProductProbabilityDistribution(self, b)

        elif isinstance(b, int):
            assert(b > 0)
            if b == 1:
                return self
            elif b > 1:
                res = ProductProbabilityDistribution(self, self)
                for i in range(1, b-1):
                    res *= self
                return res

    def __rmul__(self, b):
        if isinstance(b, int):
            assert(b > 0)
            if b == 1:
                return self
            elif b > 1:
                res = ProductProbabilityDistribution(self, self)
                for i in range(1, b-1):
                    res *= self
                return res

    def logtransform(self):
        """ Distribution of the log-transformed random variable.

        Returns
        -------

        prob_dist : ProbabilityDistribution
            The distribution of the log-transformed random variable represented
            as a `UnivariateProbabilityDistribution` object.
        
        """
        return LogTransformedProbabilityDistribution(self)

    def scaletransform(self, c):
        return ScaleTransformedProbabilityDistribution(c, self)

class LogTransformedProbabilityDistribution(ProbabilityDistribution):

    def __init__(self, orig_prob):
        super(LogTransformedProbabilityDistribution, self).__init__(orig_prob.output_dim)
        self._p = orig_prob

    def logpdf(self, x, eval_gradient=False):
        ex = np.exp(x)
        if eval_gradient:
            f, df = self._p.logpdf(ex, True)
            return x + f, 1 + df * ex
        else:
            return x +  self._p.logpdf(ex)

class ScaleTransformedProbabilityDistribution(ProbabilityDistribution):

    def __init__(self, c, orig_prob):
        super(ScaleTransformedProbabilityDistribution, self).__init__(orig_prob.output_dim)
        self._p = orig_prob
        self._scalefactor = c

    def logpdf(self, x, eval_gradient=False):
        if eval_gradient:
            f, df = self._p.logpdf(x/self._scalefactor, True)
            return f - np.log(abs(self._scalefactor)), \
                   df/self._scalefactor
        else:
            return self._p.logpdf(x/self._scalefactor) - np.log(abs(self._scalefactor))

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

class UnivariateProbabilityDistribution(ProbabilityDistribution):

    def __init__(self):
        """Probability distribution of a scalar random variable
        """
        super(UnivariateProbabilityDistribution, self).__init__(output_dim=1)

class ChiSquare(UnivariateProbabilityDistribution):
    def __init__(self, df=1):
        super(ChiSquare, self).__init__()
        self.df = df

    def logpdf(self, x, eval_gradient=False):
        k = self.df
        lpdf = (.5*k - 1)*np.log(x) - .5*x - \
               .5*k*np.log(2) - gammaln(.5*k)
        if eval_gradient:
            return lpdf, (.5*k - 1)/x - .5
        else:
            return lpdf

class Gamma(UnivariateProbabilityDistribution):
    def __init__(self, a=1, b=1):
        super(Gamma, self).__init__()
        self.a = a
        self.b = b

    def logpdf(self, x, eval_gradient=False):
        lpdf = (self.a-1)*np.log(x) - self.b*x + self.a*np.log(self.b) - gammaln(self.a)
        if eval_gradient:
            lpdf_dx = (self.a-1)/x - self.b
            return lpdf, lpdf_dx
        return lpdf

class GeneralisedInverseGaussian(UnivariateProbabilityDistribution):
    def __init__(self, a=5, b=5, p=-1):
        super(GeneralisedInverseGaussian, self).__init__()
        self.a = a
        self.b = b
        self.p = -1

    def logpdf(self, x, eval_gradient=False):
        lpdf = 0.5*self.p*np.log(self.a/self.b) + \
               (self.p-1)*np.log(x) - \
               .5*(self.a*x + self.b/x)
        if eval_gradient:
            lpdf_dx = (self.p-1)/x + .5*(self.a - self.b/x**2)
            return lpdf, lpdf_dx
        return lpdf

class Normal(UnivariateProbabilityDistribution):
    def __init__(self, loc=0, scale=1.):
        super(Normal, self).__init__()
        self.loc = loc
        self.scale = scale

    def logpdf(self, x, eval_gradient=False):
        lpdf = -0.5*(x-self.loc)**2/self.scale**2
        lpdf -= .5 * np.log(2*np.pi*self.scale**2)
        if eval_gradient:
            return lpdf, -(x-self.loc)/self.scale**2
        else:
            return lpdf

class Laplace(UnivariateProbabilityDistribution):
    def __init__(self, loc=0., scale=1.):
        super(Laplace, self).__init__()
        self.loc = loc
        self.scale = scale

    def logpdf(self, x ,eval_gradient=False):
        lpdf = -abs(x-self.loc)/self.scale
        lpdf -= np.log(2*self.scale)
        if eval_gradient:
            dx = -np.sign(x-self.loc)/self.scale
            return lpdf, -np.sign(x-self.loc)/self.scale
        else:
            return lpdf
    
class InverseGamma(UnivariateProbabilityDistribution):

    def __init__(self, a=1, b=1):
        """Inverse Gamma probability distribution
        
        Notes
        -----
        The probability density for the Inverse Gamma distribution is

        .. math::

            p(x) = \\frac{\\beta^{\\alpha}}{\\Gamma(\\alpha)}
            x^{-(\\alpha+1)} e^{-\\beta x^{-1}}

        where :math:`\\alpha` is the shape parameter and :math:`\\beta` is
        the scale parameter.

        Examples
        --------

        >>> # compare with the gamma distribution in scipy.stats
        >>> from scipy.stats import gamma
        >>> p = InverseGamma()
        >>> q = p.logtransform()  # dist. of log transformed inv. gamma r.v.
        >>> z = gamma.rvs(a=2, size=1000)
        >>> x = np.linspace(-5., 5., 100)
        >>> # plot the pdf of the log transformed inv. gamma r.v.
        >>> plt.plot(x, np.exp(q.logpdf(x)))
        >>> # plot the histogram of log(1/Z) Z ~ gamma(a, b)
        >>> plt.hist(-np.log(z), density=True, alpha=0.5)

        .. plot::

            import matplotlib.pyplot as plt
            from scipy.stats import gamma
            import numpy as np
            from pydygp.probabilitydistributions import InverseGamma
            p = InverseGamma(a=2)
            q = p.logtransform()
            z = gamma.rvs(a=2, size=1000)
            x = np.linspace(-5., 5., 100)
            plt.plot(x, np.exp(q.logpdf(x)))
            plt.hist(-np.log(z), density=True, alpha=0.5)
            plt.show()

        """        
        super(InverseGamma, self).__init__()
        self.a = a
        self.b = b

    def logpdf(self, x, eval_gradient=False):
        lpdf = -(self.a+1)*np.log(x) - self.b/x
        lpdf += self.a*np.log(self.b) - gammaln(self.a)
        if eval_gradient:
            return lpdf, -(self.a+1)/x + self.b/x**2
        else:
            return lpdf
        
        


class ExpGamma(ProbabilityDistribution):
    def __init__(self, a=1, b=1):
        self.a = a
        self.b = b

    def logpdf(self, x, eval_gradient=False):
        ex = np.exp(x)

        lpdf = self.a*x - self.b*ex
        lpdf += self.a*np.log(self.b) - gammaln(self.a)

        if eval_gradient:
            lpdf_dx = self.a - self.b*ex
            return lpdf, lpdf_dx

        return lpdf

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


class MultivariateNormal(ProbabilityDistribution):

    def __init__(self, mean, cov, jitter=True, alpha=1e-5):
        N = cov.shape[0]
        super(MultivariateNormal, self).__init__(output_dim=N) 
        
        self.mean = np.asarray(mean)
        if jitter:
            cov[np.diag_indices_from(cov)] += alpha
        L = np.linalg.cholesky(cov)
        self._L = L

    def logpdf(self, x, eval_gradient=False):
        L = self._L
        alpha = cho_solve((L, True), x - self.mean)

        log_lik = -.5 * x.dot(alpha)
        log_lik -= np.log(np.diag(L)).sum()
        log_lik -= L.shape[0] / 2 * np.log(2 * np.pi)

        if eval_gradient:
            return log_lik, -alpha
        else:
            return log_lik

    def rvs(self, size=1):
        z = np.random.normal(size=size*self.output_dim).reshape(self.output_dim, size)
        z = self._L.T.dot(z) + self.mean[:, None]
        if size == 1:
            return z.ravel()
        else:
            return z.T

    @property
    def cov(self):
        return self._L.dot(self._L.T)

    
        
