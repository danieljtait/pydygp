"""
Variational Inference
=====================
"""
import numpy as np
import matplotlib.pyplot as plt
from pydygp.probabilitydistributions import (Normal,
                                             GeneralisedInverseGaussian,
                                             Gamma)
from sklearn.gaussian_process.kernels import RBF
from pydygp.liealgebras import so
from pydygp.linlatentforcemodels import VarMLFMAdapGrad
                                             
np.random.seed(17)
np.set_printoptions(precision=3, suppress=True)

mlfm = VarMLFMAdapGrad(so(3), R=1, lf_kernels=(RBF(), ))

beta = np.row_stack(([0.]*3,
                     np.random.normal(size=3)))

x0 = np.eye(3)

# Time points to solve the model at
tt = np.linspace(0., 6., 9)

# Data and true forces
Data, lf = mlfm.sim(x0, tt, beta=beta, size=3)

# vectorise and stack the data
Y = np.column_stack((y.T.ravel() for y in Data))

logpsi_prior = GeneralisedInverseGaussian(a=5, b=5, p=-1).logtransform()
loggamma_prior = Gamma(a=2.00, b=10.0).logtransform() * mlfm.dim.K
beta_prior = Normal(scale=1.) * beta.size

fitopts = {'logpsi_is_fixed': True, 'logpsi_prior': logpsi_prior,
           'logphi_is_fixed': False,
           'g0': lf[0](tt),
           'loggamma_is_fixed': False, 'loggamma_prior': loggamma_prior,
           'beta_is_fixed': True, 'beta_prior': beta_prior,
           'beta0': beta,
           }

mapres, Eg, Covg, Eb, Covb = mlfm.varfit(tt, Y, **fitopts)
print(mapres.logphi)
print(mapres.loggamma)

ttd = np.linspace(tt[0], tt[-1], 100)
aaTrue = mlfm._component_functions(lf[0](ttd), beta, N=ttd.size)
aaMap = mlfm._component_functions(mapres.g, mapres.beta)
Eaa = mlfm._component_functions(Eg, Eb)

fig, ax = plt.subplots()
inds = [(0, 1), (0, 2), (1, 2)]
for ind in inds:
    i, j = ind
    ax.plot(ttd, aaTrue[i, j, :], '-')
    ax.plot(tt, Eaa[i, j, :], '+')
    ax.plot(tt, aaMap[i, j, :], 's')


from scipy.interpolate import interp1d
lfhat = interp1d(tt, mapres.g.ravel(), kind='cubic', fill_value='extrapolate')

for m in range(3):

    sol, _ = mlfm.sim(x0[m, :], ttd, mapres.beta, latent_forces=(lfhat,))

    fig, ax = plt.subplots()
    ax.plot(ttd, sol, 'k-', alpha=0.5)
    ax.plot(tt, Data[m], 's')

plt.show()
