"""

Kubo Oscillator
===============

This note continues on from the :ref:`basic MAP tutorial<tutorials-mlfm-ag>`
examining the Adaptive Gradient matching approximation the MLFM.

"""
import numpy as np
import matplotlib.pyplot as plt
from pydygp.liealgebras import so
from pydygp.linlatentforcemodels import MLFMAdapGrad
from pydygp.probabilitydistributions import Normal, Laplace
from sklearn.gaussian_process.kernels import RBF

np.random.seed(12345)

mlfm = MLFMAdapGrad(so(2), R=1, lf_kernels=(RBF(), ))


x0 = np.array([1., 0.])
beta = np.array([[0., ], [1., ]])

ttd = np.linspace(0., 5., 100)
data, lf = mlfm.sim(x0, ttd, beta=beta)

tt = ttd[::10]
Y = data[::10, :]

mapres = mlfm.fit(tt, Y.T.ravel(),
                  logpsi_is_fixed=True,
                  beta_is_fixed=True, beta0=beta)
gpred = mlfm.predict_lf(ttd)

fig, ax = plt.subplots()
ax.plot(ttd, lf[0](ttd), 'k-', alpha=0.3)
ax.plot(ttd, gpred[0], 'C0-')
print(mapres.optimres.fun)
########################################################
#
# :math:`\beta` free
# ==================
#
mapres2 = mlfm.fit(tt, Y.T.ravel(),
                   beta0=beta, logpsi_is_fixed=True)
gpred2 = mlfm.predict_lf(ttd)
ax.plot(ttd, gpred2[0], 'r-')
print(mapres2.optimres.fun)
########################################################
#
# So whats happened? The latent force looks like it has
# collapsed to a constant valued function. Lets plot just
# function itself to get an idea what's going on
fig2, ax2 = plt.subplots()
ax2.plot(ttd, gpred2[0], 'r-')

from scipy.interpolate import interp1d
ginterp = interp1d(ttd, gpred2[0],
                   kind='cubic', fill_value='extrapolate')

fig3, ax3 = plt.subplots()
data2, _ = mlfm.sim(x0, ttd,
                    beta=mapres2.beta, latent_forces=(ginterp, ))
ax3.plot(ttd, data2, 'C0-')
ax3.plot(tt, Y, 'o')

beta_prior = Normal() * Normal()

mapres3 = mlfm.fit(tt, Y.T.ravel(),
                   beta0=beta,
                   logpsi_is_fixed=True,
                   beta_prior = beta_prior)
print(mapres3.optimres.fun)
print(mapres2.beta)
print(mapres3.beta)
gpred3 = mlfm.predict_lf(ttd)

fig4, ax4 = plt.subplots()
ax4.plot(ttd,
         mapres3.beta[0, 0] + mapres3.beta[1, 0]*gpred3[0],
         'C0--')
ax4.plot(ttd, lf[0](ttd), 'k-', alpha=0.3)

plt.show()

