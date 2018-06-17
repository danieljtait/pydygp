import numpy as np
import matplotlib.pyplot as plt
from pydygp.linlatentforcemodels import MLFM
from pydygp.gaussianprocesses import GaussianProcess

np.random.seed(5)

Lx = np.array([[0., 0., 0.],
               [0., 0.,-1.],
               [0., 1., 0.]])

Ly = np.array([[0., 0.,-1.],
               [0., 0., 0.],
               [1., 0., 0.]])

times = np.linspace(0., 4., 8)

# make the basic mlfm model
mlfm = MLFM.ns([Lx, Ly], order=1)

# inital gaussian process approx
#  - ind. gps, for each dimension
x0_gps = [GaussianProcess('sqexp') for k in range(3)]

# latent force gaussian processes
g_gps = [GaussianProcess('sqexp', kpar=[1., 1.]), ]

# simulate some data
y, gtrue, ttd, yd = mlfm.sim([0., 0., 1.], times, gps=g_gps, return_gp=True)

mlfm.em_fit(times,
            y.T.ravel(),
            x0_gps=x0_gps, g_gps=g_gps,
            ifix=4)

# making xcov
from scipy.linalg import block_diag

ttf = mlfm.em.comp_times
x0k_covs = [gp.kernel.cov(ttf[:, None]) for gp in mlfm.em.x0_gps]
g_covs = [gp.kernel.cov(ttf[:, None]) for gp in mlfm.em.g_gps]

# Jitter the covariance matrices
# to avoid singular matrices
for c in g_covs:
    c += np.diag(1e-4*np.ones(c.shape[0]))

for c in x0k_covs:
    c += np.diag(1e-4*np.ones(c.shape[0]))

mlfm.em.xcov = block_diag(*x0k_covs)
mlfm.em.xcov_invs = [np.linalg.inv(c) for c in x0k_covs]

mlfm.em.gcov = block_diag(*g_covs)
mlfm.em.gprior_inv_cov = block_diag(*[np.linalg.inv(c) for c in g_covs])

L = np.eye(mlfm.em.vecy.size)*1e6 + np.dot(mlfm.em.data_map, mlfm.em.data_map.T)*mlfm.em.beta
mlfm.em.data_covar = np.linalg.inv(L)



fig, ax = plt.subplots()
ax.plot(ttd, gtrue[:, 0], 'k-', alpha=0.2)
ax.plot(ttf, mlfm.em.fit(liktol=1e-2), 'b+')
ax.plot(ttf, mlfm.em.fit(liktol=1e-3), 'r+')
ax.plot(ttf, mlfm.em.fit(liktol=1e-4), 'g+')

"""
m = m[:mlfm.em.dim.N]
sd = np.sqrt(np.diag(c))[:mlfm.em.dim.N]

fig, ax = plt.subplots()
ax.fill_between(ttf, m + 2*sd, m-2*sd, alpha=0.2)
ax.plot(ttf, m)
ax.plot(ttd, yd, alpha=.2)
"""

plt.show()
