
import numpy as np
from scipy.special import gammaln

a = 1.1
b = 0.5

def log_fx(x):
    return (a-1)*np.log(x) - b*x + a*np.log(b) - gammaln(a)

def log_fx_grad(x):
    return (a-1)/x - b

def fx(x):
    return np.exp(log_fx(x))


def fy(y):
    ey = np.exp(y)
    return ey*fx(ey)

def log_fy(y):
    return y + log_fx(np.exp(y))

def log_fy_grad(y):
    ey = np.exp(y)
    return 1 + log_fx_grad(ey)*ey

y = np.random.normal()

eps = 1e-6

#print((log_fy(y+eps) - log_fy(y))/eps)
#print(log_fy_grad(y))

from probabilitydistributions import (ProbabilityDistribution,
                                      Gamma,
                                      ExpGamma,
                                      ExpGeneralisedInvGaussian)

p = ExpGamma(a=1.5)
q = Gamma(a=1.5).logtransform()

print(p.logpdf(-1.4, True))
print(q.logpdf(-1.4, True))

q1 = p * 3
q2 = 3 * p

x = np.random.uniform(size=3)

print(q1.logpdf(x, True))
print(q2.logpdf(x, True))

#from scipy.integrate import quad
#print(quad(fy, -np.inf, 20.))

