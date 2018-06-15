import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.linalg import block_diag
from pydygp.linlatentforcemodels import MLFM_MH_NS as MLFM

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

q = .5
a = .3

def g(t):
    return 2*q*np.cos(2*t)

A0 = np.array([[0., 1.],
               [-a, 0.]])
A1 = np.array([[0., 0.],
               [1., 0.]])

def dXdt(X, t):
    return np.dot(A0 + A1*g(t), X)

x0 = np.array([0., 1.])
#tt = np.linspace(0., 10.5, 100)
tt = np.sort(np.random.uniform(0., 10.5, size=100))
sol = odeint(dXdt, x0, tt)

n = 15
times = tt[0:tt.size:n]
Y = sol[0:tt.size:n, :]
print(Y.shape)

mlfm = MLFM([A0, A1])
mlfm.time_interval_setup(times, h=None)
mlfm.data_Y = Y

w1 = 100000*np.ones(times.size)#(times+1)**2
w2 = 100000*np.ones(times.size)#20*(times+1)**2
L1 = np.diag(w1)
L2 = np.diag(w2)
mlfm.Lambda_cur = block_diag(L1, L2)

mlfm.phi_setup()
mlfm.x_gp_setup()
mlfm.g_gp_setup()
mlfm.phi_init([[.1, .5], [.1, .5]])
mlfm.psi_cur = [[1., 1.], ]

# solve ODE for true values at mlfm complete time points
xsol = odeint(dXdt, x0, mlfm.comp_times)

# set x_cur in mlfm to true solutions
mlfm.x_cur = [x for x in xsol.T]
mlfm.g_cur = [g(mlfm.comp_times), ]

import nssolve
from scipy.stats import multivariate_normal
mlfm.myop = nssolve.QuadOperator(method='single_fp',
                                 fp_ind=9,
                                 intervals=mlfm.intervals,
                                 K=mlfm.dim.K, R=mlfm.dim.R,
                                 struct_mats=mlfm.struct_mats,
                                 is_x_vec=False)
                                   

mg, Cg = mlfm.g_cond_meanvar()
mx, Cx = mlfm.x_cond_meanvar()

rx = []
rg = []
nsim = 500
for nt in range(nsim):
    mg, Cg = mlfm.g_cond_meanvar()
    gnew = multivariate_normal.rvs(mean=mg, cov=Cg)

    mlfm.g_cur = [gnew, ]

    mx, Cx = mlfm.x_cond_meanvar()
    xnew = multivariate_normal.rvs(mean=mx, cov=Cx)

    mlfm.x_cur = [x for x in xnew.reshape(mlfm.dim.K, mlfm.dim.N)]
    if nt > nsim / 2:
        rg.append(gnew)
        rx.append(xnew[:mlfm.dim.N])

rg = np.array(rg)
rx = np.array(rx)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(mlfm.comp_times, rg.T, 'k+', alpha=0.2)
ttd = np.linspace(tt[0], tt[-1], 100)
ax.plot(ttd, g(ttd), 'r-')

fig2 = plt.figure()
ax = fig2.add_subplot(111)
ax.plot(mlfm.comp_times, rx.T, 'k+', alpha=0.2)
#ax.plot(mlfm.comp_times, xsol[:, 0], 'r-')
ax.plot(times, Y[:, 0], 'rs')

plt.show()

"""
fig = plt.figure()
ax = fig.add_subplot(111)
ttd = np.linspace(mlfm.comp_times[0],
                  mlfm.comp_times[-1],
                  100)
ax.plot(ttd, g(ttd), 'k-', alpha=0.2)
#ax.plot(mlfm.comp_times, mg, '+')
sdg = np.sqrt(np.diag(Cg))
ax.plot(mlfm.comp_times, mg, '-.')
ax.fill_between(mlfm.comp_times,
                mg + 2*sdg,
                mg - 2*sdg,
                alpha=0.2)


fig2 = plt.figure()
ax = fig2.add_subplot(111)
ax.plot(mlfm.comp_times, xsol, '-', alpha=0.2)
ax.plot(times, Y, 'ks')
ax.plot(mlfm.comp_times, mx.reshape(mlfm.dim.K, mlfm.dim.N).T, '+')

plt.show()
"""

""" 





mean, km, K, cov = mlfm.x_cond_meanvar()
mean = mean.reshape(2, mlfm.dim.N).T
km = km.reshape(2, mlfm.dim.N).T
cov = np.dot(K, np.dot(cov, K.T))
fig = plt.figure()
ax = fig.add_subplot(111)
#ax.plot(mlfm.comp_times, km, '+-.')
_x = mean.T.ravel()
for nt in range(1):
    _x = np.dot(K, _x)
Km = _x.reshape(2, mlfm.dim.N).T

sd = np.sqrt(np.diag(cov))

#ax.plot(mlfm.comp_times, mean, 'o')
ax.plot(tt, sol, '-')
ax.plot(mlfm.comp_times, Km, 'k-.')
ax.fill_between(mlfm.comp_times,
                Km[:, 0] - 2*sd[:mlfm.dim.N],
                Km[:, 0] + 2*sd[:mlfm.dim.N], alpha=0.2)
ax.fill_between(mlfm.comp_times,
                Km[:, 1] - 2*sd[mlfm.dim.N:],
                Km[:, 1] + 2*sd[mlfm.dim.N:], alpha=0.2)
ax.plot(times, Y, 's')
plt.show()
"""
