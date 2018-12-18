"""

Gibbs Sampling in the Product Model
===================================
"""
import numpy as np
import matplotlib.pyplot as plt
from pydygp.liealgebras import so
from pydygp.linlatentforcemodels import GibbsMLFMAdapGrad
from pydygp.probabilitydistributions import (Normal, Laplace)
from sklearn.gaussian_process.kernels import RBF



N_manifolds = 2

mlfm1 = GibbsMLFMAdapGrad(so(2), R=1, lf_kernels=[RBF(), ])
mlfm2 = GibbsMLFMAdapGrad(so(2), R=1, lf_kernels=[RBF(), ])

mmlfm = mlfm1 * mlfm2


x0_0 = [1., 0.]
x0_0_1 = [0., 1.]
x0_1 = [0., 1.]
beta_0 = np.array([[0., ], [1, ]])
beta_1 = np.array([[-0.3, ], [1, ]])

beta = (beta_0, beta_1, )

tt = np.linspace(0., 5., 7)

# make the data by hand
Y0, lf = mlfm1.sim([x0_0, x0_0_1], tt, beta=beta[0], size=2)
Y1, _ = mlfm2.sim(x0_1, tt, beta=beta[1], size=1, latent_forces=lf)

Ydata = [Y0, (Y1, )]

fitopts = {'loggamma_is_fixed': True}
fitopts['logpsi_is_fixed'] = True
fitopts['beta_is_fixed'] = True
fitopts['logphi_is_fixed'] = True
fitopts['logtau_is_fixed'] = True
for i in range(N_manifolds):
    fitopts['beta_{}_init'.format(i)] = beta[i]

# new style of training data
# idea being that Ydata[q][m]
# returns the mth observation from the qth mlfm model
#_Y = [[y] for y in Y]
#Ydata = [[y for y in item] for item in _Y]

xrv, Eg, rvs = mmlfm.gibbs_sample(tt, Ydata, **fitopts)

fig, ax = plt.subplots()
ax.plot(tt, lf[0](tt), '--')
ax.plot(tt, mlfm1.mapres.g, '+')
ax.plot(tt, Eg, '-o')
ax.plot(tt, rvs['g'].T, 'k+', alpha=0.2)

fig, ax = plt.subplots()
ax.plot(tt, Y0[1], '+')

#x = xrv[0][:, 1].reshape((2, tt.size)).T
ax.plot(tt, xrv[0][1], '-')

plt.show()
