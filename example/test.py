import numpy as np
from pydygp.linlatentforcemodels import MLFM_AdapGrad
from pydygp.kernels import Kernel,  GradientKernel
from pydygp.gaussianprocesses import GaussianProcess, GradientGaussianProcess
from scipy.integrate import odeint
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

A0 = np.array([[0., 0., 0.],
               [0., 0., 1.],
               [0.,-1., 0.]])
A1 = np.array([[0., 0.,-1.],
               [0., 0., 0.],
               [1., 0., 0.]])
A2 = np.array([[0., 1., 0.],
               [-1., 0., 0.],
               [0., 0., 0.]])

mlfm = MLFM_AdapGrad([A0, A1, A2])


def g1(t):
    return np.cos(t)*np.exp(-t)

def g2(t):
    return np.sin(t)

def dXdt(x, t):
    return np.dot(A0 + A1*g1(t) + A2*g2(t), x)

tt = np.linspace(0., 2., 8)
x0 = [0., 0., 1.]
sol = odeint(dXdt, x0, tt)

"""
Setup the adaptive gradient matching model
"""
mlfm.setup(tt)

x_kernel_hyperpars = ([1.7, 1.3],
                      [1.0, 0.9],
                      [1.1, 1.0], )
x_kernels = [GradientKernel.SquareExponKernel(par, dim=1)
             for par in x_kernel_hyperpars]
x_gps = [GradientGaussianProcess(kern) for kern in x_kernels]

mlfm.x_gps = x_gps
mlfm._update_x_cov_structure()

g_kernel_hyperpars = ([1.1, 0.5],
                      [1.3, 1.0],)
g_kernels = [Kernel.SquareExponKernel(par, dim=1)
             for par in g_kernel_hyperpars]

mlfm.setup_latentforce_gps(g_kernels)
mlfm._update_g_cov_structure()


vecx = sol.T.ravel()
vecg = np.concatenate([g1(tt), g2(tt)])

ic = mlfm.model_x_dist(vecg)

from scipy.linalg import solve_triangular
# x prior inv cov
x_prior_inv_cov = [solve_triangular(L.T,
                                    solve_triangular(L, np.eye(tt.size), lower=True))
                       
                   for L in mlfm._x_cov_chols]
from scipy.linalg import block_diag
x_prior_inv_cov = block_diag(*x_prior_inv_cov)

ic += x_prior_inv_cov
c = np.linalg.inv(ic)

k = 0
uk = mlfm.x_flowk_rep(k, vecg)
vk, vk0 = mlfm.g_flowk_rep(k, vecx)

print(sum(ukj*xj for ukj, xj in zip(uk, vecx.reshape(3, tt.size))))
print(sum(vkr*gr for vkr, gr in zip(vk, vecg.reshape(2, tt.size))) + vk0)


"""
from scipy.optimize import minimize
def objfunc(g):
    expr1 = mlfm.log_model_likelihood(vecx, g)
    expr2 = mlfm.log_prior(vecx=None, vecg=g)
    return -(expr1 + expr2)

g = np.zeros(vecg.size)

res = minimize(objfunc, g)
print(vecg.reshape(2, tt.size).T)
print(res.x.reshape(2, tt.size).T)
print(res.message)
"""
