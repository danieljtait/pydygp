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
tt = np.linspace(0., 10.5, 100)
sol = odeint(dXdt, x0, tt)

n = 12
times = tt[0:tt.size:n]
Y = sol[0:tt.size:n, :]

mlfm = MLFM([A0, A1])
mlfm.time_interval_setup(times, h=.2)
mlfm.phi_setup()
mlfm.x_gp_setup()
mlfm.phi_init([[.1, .5], [.1, .5]])

# 
mlfm.g_cur = [g(mlfm.comp_times), ]
mlfm.data_Y = Y

w1 = 1000*np.ones(times.size)#(times+1)**2
w2 = 1000*np.ones(times.size)#20*(times+1)**2
L1 = np.diag(w1)
L2 = np.diag(w2)
mlfm.Lambda_cur = block_diag(L1, L2)

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
