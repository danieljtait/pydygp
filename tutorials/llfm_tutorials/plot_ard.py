"""
Automatic Relevance Determination
=================================
"""
import numpy as np
import matplotlib.pyplot as plt
import pydygp.liealgebras
from pydygp.linlatentforcemodels import MLFMAdapGrad
from sklearn.gaussian_process.kernels import RBF
from scipy.integrate import odeint
np.random.seed(9)

##############################################################################
#
# We are going to set up two latent force models, the first is the simple
# random harmonic oscillator on :math:`\mathbb{R}^2` which we have discussed in
# previous tutorials and we recall is given by the ODE :math:`\dot{X}(t) = A(t)X(t)`
# with
#
# .. math::
#
#     \begin{bmatrix} 0 & -g_1(t) \\ g_1(t) \\ \end{bmatrix}.
#
# We then extend this model, we are also going to make the force :math:`g_2(t)`
# a small so that model two provides a small pertubation of the original model
# (this is not necessary for fitting but will make visualisation here more
# appealing).

so2 = pydygp.liealgebras.so(2)
I = np.eye(2)


k1 = 1*RBF()

eps = 5e-1
k2 = eps*RBF()

mlfm1 = MLFMAdapGrad(so2, lf_kernels=(k1, ))
mlfm2 = MLFMAdapGrad((*so2, I), lf_kernels=(k1, k2))

tt = np.linspace(0., 7., 100)

Y1, gf = mlfm1.sim([1., 0.], tt)
#Y2, gf = mlfm2.sim([1., 0.], tt)

g2 = lambda t: np.cos(2*np.pi*t)
def dXdt(X, t):
    At = so2[0]*gf[0](t) + g2(t)*I
    return At.dot(X)

inds = np.linspace(0, tt.size-1, 10, dtype=np.intp)
Y2 = odeint(dXdt, [1., 0.], tt)

fig, ax = plt.subplots()
ax.plot(*Y1.T, '-', alpha=0.5, label='model 1')
ax.plot(*Y2.T, '-', alpha=0.5, label='model 2')
ax.legend()
ax.set_aspect('equal')

plt.show()
