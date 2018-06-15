import numpy as np
from pydygp.linlatentforcemodels import MLFM_MH_AdapGrad as mlfm

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
"""
Model setup 
"""

# 3 basis matrices of the Lie algebra so(2)

Lx = np.array([[0., 0., 0.],
               [0., 0.,-1.],
               [1., 0., 0.]])

Ly = np.array([[0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.]])

Lz = np.array([[0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.]])

mlfm_mod = mlfm([np.zeros((3, 3)), Lx, Ly, Lz])

#mlfm_mod.add_latent_states(datatimes=data_times,
#                           aug_times="linspace"
#tt = np.linspace(0., 5., 4)
#mlfm_mod.setup(tt)#, aug_times='linspace')
#for gp in mlfm_mod.x_gps:
#    print(gp)


from scipy.optimize import root
from scipy.integrate import odeint
def backwards_euler_step(y0, t0, h, f, *args, **kwargs):

    def _objfunc(y):
        err = y - y0 - h*f(y, t0+h, *args, **kwargs)
        return err

    res = root(_objfunc, x0=y0 + h*f(y0, t0, *args, **kwargs))
    if res.success:
        return res.x

def rk4_step(y0, t0, h, f, *args, **kwargs):

    k1 = h*f(y0, t0, *args, **kwargs)
    k2 = h*f(y0 + 0.5*k1, t0 + 0.5*h, *args, **kwargs)
    k3 = h*f(y0 + 0.5*k2, t0 + 0.5*h, *args, **kwargs)
    k4 = h*f(y0 + k3, t0 + h, *args, **kwargs)

    return y0 + (k1 + 2*k2 + 2*k3 + k4)/6.
        
def f(y, t):
    return np.cos(t)*y

y0 = 1.
t0 = 0.
h = 0.2

tt = np.linspace(0., .15, 10)

sol = [y0]
sol3 = [y0]
for ta, tb in zip(tt[:-1], tt[1:]):
    sol.append(backwards_euler_step(sol[-1],
                                    ta,
                                    tb-ta,
                                    f))
    sol3.append(rk4_step(sol3[-1],
                         ta,
                         tb-ta,
                         f))

sol = np.array(sol)
sol3 = np.array(sol3)
sol2 = odeint(f, y0, tt)
#print(sol[:, None])
#print(sol2)
#print(sol3[:, None])

# rk4 dense time points
ttdense = np.concatenate([[ta, 0.5*(ta+tb)]
                          for ta, tb in zip(tt[:-1], tt[1:])])
ttdense = np.concatenate((ttdense, [tt[-1]]))


u = lambda t: np.interp(t, ttdense, np.cos(ttdense))

def dXdt(y, t, u):
    return (1+u(t))*y

print(u(0.5))

def _sim_gp_interpolators(tt, gps, return_gp):
    # rk4 dense time points
    ttdense = np.concatenate([[ta, 0.5*(ta+tb)]
                              for ta, tb in zip(tt[:-1], tt[1:])])
    ttdense = np.concatenate((ttdense, [tt[-1]]))

    rvs = [gp.sim(ttdense[:, None]) for gp in gps]

    gs = [lambda t: np.interp(t, ttdense, rv) for rv in rvs]

    if return_gp:
        return gs, np.column_stack(rvs), ttdense
    else:
        return gs, None, None


def sim(x0, tt, struct_mats, gs=None, gps=None, return_gp=False):

    # dimensional rationality checks
    if gs is not None:
        assert(isinstance(gs, (tuple, list)))
        assert(len(gs) == len(struct_mats)-1)

    if gs is None and gps is not None:
        gs, gval, ttd = _sim_gp_interpolators(tt, gps, return_gp)

    def dXdt(X, t):
        # constant part of flow
        A0 = struct_mats[0]

        # time dependent part
        At = sum([Ar*gr(t) for Ar, gr in zip(struct_mats[1:], gs)])
        
        return np.dot(A0 + At, X)

    sol = odeint(dXdt, x0, tt)

    if return_gp:
        return sol, gval, ttd
    else:
        return sol
    
from pydygp.kernels import Kernel
from pydygp.gaussianprocesses import GaussianProcess

kse = Kernel.SquareExponKernel()
gp = GaussianProcess(kse)
gp2 = GaussianProcess(kse)

tt = np.linspace(0., 3., 25)
ttdense = np.linspace(0., 3., tt.size*5)

A0 = np.zeros((3, 3))
A1 = np.array([[0.,-1.,-1.],
               [1., 0., 0.],
               [1., 0., 0.]])
A2 = np.array([[0., 0., 0.],
               [0., 0.,-1.],
               [0., 1., 0.]])

x0 = np.random.normal(size=3)
x0 /= np.linalg.norm(x0)
sol, g, ttd = sim(x0, tt, [A0, A1, A2], gps=[gp, gp2], return_gp=True)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)

r_tt = tt[::5]
r_sol = sol[::5, ]

ax.plot(tt, sol, '+')
ax.plot(r_tt, r_sol, 'o')

fig2 = plt.figure()
ax = fig2.add_subplot(111)
ax.plot(ttd, g, '+')

plt.show()
