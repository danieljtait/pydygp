import numpy as np
import scipy.stats
from collections import namedtuple
import matplotlib.pyplot as plt

Proposal = namedtuple('Proposal', 'rvs')
Prior = namedtuple('Prior', 'pdf logpdf')

# Lightweight proposal and prior distributions
# note that the proposal should return the tuple
#
#     proposal.rvs(x) = (xnew, q(x)/q(xnew))
#
prop = Proposal(lambda x: (np.random.normal(loc=x, scale=1.), 1))
prior = Prior(lambda x: np.exp(sum(scipy.stats.uniform.logpdf(x, loc=[0, 0], scale=[1, 2]))),
              lambda x: sum(scipy.stats.uniform.logpdf(x, loc=[0, 0], scale=[1, 2])))


X = [[0.5, 1.]]
lp = prior.logpdf(X[-1])
ac = 0
for nt in range(5000):
    xnew, qr = prop.rvs(X[-1])
    lpnew = prior.logpdf(xnew)
    A = np.exp(lpnew - lp)*qr
    if np.random.uniform() <= A:
        X.append(xnew)
        lp = lpnew
        ac += 1
    else:
        X.append(X[-1])
X = np.array(X)
print(ac/X.shape[0])

X = X[::10,]
print(X.shape)

plt.plot(X)
plt.show()
