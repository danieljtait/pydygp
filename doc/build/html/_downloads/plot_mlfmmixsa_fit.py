"""
Fitting of the MLFM-MixSA Model
===============================
"""
import numpy as np
import matplotlib.pyplot as plt
from pydygp.linlatentforcemodels import MLFMMixSA
from sklearn.gaussian_process.kernels import RBF
from pydygp.liealgebras import so
from collections import namedtuple
np.set_printoptions(precision=3, suppress=True)

mlfm = MLFMMixSA(so(3), R=1, order=10, lf_kernels=[RBF(), ])

x0 = np.eye(3)  # initial conditions the std. basis vectors of R3
N_outputs = 3

beta = np.random.normal(size=6).reshape(2, 3)
beta /= np.linalg.norm(beta, axis=1)[:, None]

tt = np.linspace(0., 6., 10)
Data, gtrue = mlfm.sim(x0, tt, beta=beta, size=N_outputs)

mlfm._setup_times([tt]*N_outputs, h=.25)

# set the fixed points
_ifix = np.linspace(0, tt.size-1, 5, dtype=np.intp)[1:-1]
ifix = [mlfm.data_inds[0][i] for i in _ifix]

import autograd.numpy as anp
Normal = namedtuple('Normal', 'logpdf')
betaprior = Normal(lambda x: -0.5 * anp.sum(x**2) - \
                   0.5 * x.size * np.log(2 * np.pi * 1.))



init_opts = {'g0': gtrue[0](mlfm.ttc),
             'g_is_fixed': False,
             'beta_is_fixed': True,
             'beta0': beta,
             'mu_ivp_is_fixed': True,
             'beta_prior': betaprior
             }

optim_opts = {
    'options': {'disp': True,
                'maxiter': 20}
    }

mlfm.fit([(tt, Y) for Y in Data],
         ifix,
         max_nt=10,
         optim_opts=optim_opts,
         verbose=True,
         **init_opts)

mu_ivp = np.array([
    np.dstack([Y[_ifx, :] for Y in Data])
    for _ifx in _ifix])
mu_ivp = mu_ivp[:, 0, ...]


pi = np.ones(len(ifix)) / len(ifix)
r = mlfm._get_responsibilities(pi, gtrue[0](mlfm.ttc), beta, mu_ivp, 1000)

fig, ax = plt.subplots()
ax.plot(tt, r[0])

fig2, ax2 = plt.subplots()
ax2.plot(tt, mlfm.g_[mlfm.data_inds[0]], '+')
ax2.plot(mlfm.ttc, gtrue[0](mlfm.ttc), 'k-', alpha=0.5)

layer = mlfm._forward(
    gtrue[0](mlfm.ttc),
    #mlfm.g_,
    beta,
    mlfm.mu_ivp_[0],
    mlfm._ifix[0])
print(layer.shape)
fig3, ax3 = plt.subplots()
#ax3.plot(mlfm.ttc, layer[..., 0], 'k-', alpha=0.3)
ax3.plot(mlfm.ttc, layer[:, 0, 0], 'b-.')
#ax3.plot(tt, layer[mlfm.data_inds[0], :, 0], 'b-.')
ax3.plot(tt, Data[0][:, 0], 's')
ax3.set_ylim((-1.1, 1.1))

fig4, ax4 = plt.subplots()
Ar = [sum(brd*Ld for brd, Ld in zip(br, mlfm.basis_mats))
      for br in beta]
Ar_ = [sum(brd*Ld for brd, Ld in zip(br, mlfm.basis_mats))
       for br in mlfm.beta_]

ax4.plot(mlfm.ttc, Ar[0][0, 1] + Ar[1][0, 1]*gtrue[0](mlfm.ttc), 'k-')
ax4.plot(mlfm.ttc, Ar_[0][0, 1] + Ar_[1][0, 1]*mlfm.g_, 'C0-')

plt.show()
