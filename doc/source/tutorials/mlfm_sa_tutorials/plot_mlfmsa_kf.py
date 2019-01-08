"""

.. _tutorials-mlfmsa-motiv-part1:

Approximate Density
===================

When using the method of successive approximation we construct
a regression model 

.. math::

   p(\\mathbf{x} \\mid \\mathbf{g}, \\boldsymbol{\\beta})
   = \\mathcal{N}(
   \\mathbf{x}
   \\mid \\mathbf{P}^{M} \\boldsymbol{\\mu}_0,
   \\alpha \\mathbf{I})
   
The idea is to construct an approximation to this density by
introduction each of the successive approximations

.. math::

   \\mathbf{z}_{i} = \\mathbf{P}\\mathbf{z}_{i-1},

the idea being that knowing the complete set of approximations
:math:`\{ z_0,\ldots,z_M\}` we can solve for the latent variables
by rearranging the linear equations, instead of manipulating the
polynomial mean function.

For this conversion to work we need to introduce a regularisation
parameter :math:`\lambda > 0` and then define :math:`\mathcal{N}(
\\mathbf{z}_{i} \\mid \\mathbf{P}\\mathbf{z}_{i-1}, \\lambda
\\mathbf{I})`, once we do this we can write log-likelihood of the
state variables

.. math::

   \\log = -\\frac{\lambda}{2} \\sum_{i=1}^{M}
   \\left(
   \\mathbf{z}_{i-1}^{\\top}\\mathbf{P}^{\\top}\\mathbf{P}\\mathbf{z}_{i-1}
   - 2\\mathbf{z}_{i}^{\\top}\\mathbf{P}\\mathbf{z}_{i-1}
   \\right)

Now the matrices :math:`\\mathbf{P}` are linear in the parameters
which means that after vectorisation they can be represented as

.. math::

   \\operatorname{vec}(\\mathbf{P}) = \\mathbf{V}\\mathbf{g} + \\mathbf{v}_0

.. note::

   The matrices :math:`\\mathbf{P}` and their affine representations are
   most easily written compactly using kronecker products, unfortunately
   these are not necessarily the best computational representations and
   there is a lot here that needs refining.

Linear Gaussian Model
---------------------
We take a break in the model to now discuss how to start putting some
of the ideas discussed above into code. For the Kalman Filter we are
going to use the code in the
`PyKalman package <https://pykalman.github.io/>`_, but hacked a little
bit to allow for filtering and smoothing of independent sequences
with a common transition matrix.
"""
import numpy as np
from pydygp.liealgebras import so
from sklearn.gaussian_process.kernels import RBF
from pydygp.linlatentforcemodels import MLFMSA
np.random.seed(123)

mlfm = MLFMSA(so(3), R=1, lf_kernels=[RBF(), ], order=10)
beta = np.array([[0.1, 0., 0.],
                 [-0.5, 0.31, 0.11]])
tt = np.linspace(0., 6., 7)
x0 = np.eye(3)
Data, g = mlfm.sim(x0, tt, beta=beta, size=3)

######################################
# Expectation Maximisation
# ------------------------
#
# So we have introduced a large collection of unintersting latent variables,
# the set of successive approximations :math:`\{ z_0, \ldots, z_M \}`, and
# so we need to integrate them out. If we define the statistics
#
# .. math::
#
#    \boldsymbol{\Psi}_0 = \sum_{i=1}^{M} \langle \mathbf{z}_{i-1}
#    \mathbf{z}_{i-1}^{\top} \rangle_{q(Z)}, \quad
#    \boldsymbol{\Psi}_1 = \sum_{i=1}^{M} \langle \mathbf{z}_{i}
#    \mathbf{z}_{i-1}^{\top} \rangle_{q(Z)}
#
# Then the objective function of the `M-step` becomes
#
# .. math::
#
#    Q(\mathbf{g}, \mathbf{g}^{old}) =
#    -\frac{1}{2} \mathbf{g}^{\top}
#    \left( \mathbf{V}^{\top}
#    (\boldsymbol{\Psi}_0 \otimes \mathbf{I}_{NK})\mathbf{V} +
#    \lambda^{-1} \mathbf{C}^{-1} \right)\mathbf{g} - 2

########################################################
#
# More sensible place to start -- the Kalman Filter performs the numerical integration
#
from pydygp.linlatentforcemodels import KalmanFilter

mlfm._setup_times(tt, h=None)
#ifx = mlfm.ttc // 2 # index left fixed by the Picard iteration

ifx = 0

A = mlfm._K(g[0](mlfm.ttc), beta, ifx)

init_conds = np.array([y[ifx, :] for y in Data])

Ndata = tt.size

# array [m0, m1, m2] with m0 = np.kron(Data[0][ifx, :], ones)
init_vals = np.kron(init_conds, np.ones(Ndata)).T
init_state_mean = np.kron(init_conds, np.ones(mlfm.dim.N)).T
final_vals = np.column_stack([y.T.ravel() for y in Data])

X = np.ma.zeros((mlfm.order, ) + init_vals.shape)  # data we are going to give to the KalmanFilter
X[0, ...] = init_vals
X[1, mlfm.order-1, ...] = np.ma.masked  # mask these values -- we have no data
X[mlfm.order-1, ...] = final_vals

NK = mlfm.dim.N*mlfm.dim.K
#observation_matrices = np.array([np.eye(NK)]*3)
C = np.zeros((Ndata*3, mlfm.dim.N*mlfm.dim.K))
_inds = np.concatenate([mlfm.data_inds[0] + k*mlfm.dim.N
                        for k in range(mlfm.dim.K)])
C[np.arange(Ndata*mlfm.dim.K), _inds] += 1
observation_matrices = np.array([C, ]*3)

kf = KalmanFilter(initial_state_mean=init_state_mean,
                  initial_state_covariance=np.eye(NK)*1e-5,
                  observation_offsets=np.zeros((mlfm.order, Ndata*3, mlfm.dim.K)),
                  observation_matrices=observation_matrices,
                  transition_matrices=A,
                  transition_covariance=np.eye(NK)*1e-5,
                  transition_offsets=np.zeros(init_vals.shape),
                  n_dim_state=NK,
                  n_dim_obs=Ndata*3)

means, covs, k_gains = kf.smooth(X)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
for i, mean in enumerate(means):
    # unvectorise the column
    m = mean[:, 0].reshape((mlfm.dim.K, mlfm.dim.N)).T
    ax.plot(mlfm.ttc, m, 'k-', alpha=(i+1)/mlfm.order)
ax.plot(tt, Data[0], 'ks')

############################################################################
#
# So the linear model seems to be performing the forward iteration in a
# reasonable way. The next challenge is to try and invert this for the
# conditional distribution.
#
# The relevant objective function is
#
# .. math::
#
#    \left(
#    \operatorname{vec}(\mathbf{P})^{\top}
#    \left(\boldsymbol{\Psi}_0 \otimes \lambda \cdot \mathbf{I} \right)
#    \operatorname{vec}(\mathbf{P})
#    + \mathbf{g}^{\top}\mathbf{C}_g^{-1}\mathbf{g}\right)
#    - 2 \lambda \operatorname{vec}(\boldsymbol{\Psi}_1)^{\top}
#    \operatorname{vec}(\mathbf{P})
#
# So the first thing we need is a function that constructs these statistics
def _get_kf_statistics(X, kf):
    """ Gets
    """
    # the mean, cov and kalman gain matrix
    means, covs, kalman_gains = kf.smooth(X)
    # pairwise cov between Cov{ z[i], z[i-1]
    # note pairwise_covs[0] = 0  - it gets ignored
    pairwise_covs = kf._smooth_pair(covs, kalman_gains)

    S0 = 0.
    for m, c in zip(means[:-1], covs[:-1]):
        S0 += c + \
              (m[:, None, :] * m[None, ...]).transpose((2, 0, 1))
    S1 = 0.
    for i, pw in enumerate(pairwise_covs[1:]):
        S1 += pw + \
              (means[i+1][:, None, :] * \
               means[i][None, ...]).transpose((2, 0, 1))

    return S0.sum(0), S1.sum(0)

#############################################################################
#
# Now we need a function that takes those created statistics and turns
# returns an estimate of the latent forces

from scipy.linalg import block_diag, cho_solve
def kron_A_N(A, N):  # Simulates np.kron(A, np.eye(N))
    m,n = A.shape
    out = np.zeros((m,N,n,N),dtype=A.dtype)
    r = np.arange(N)
    out[:,r,:,r] = A
    out.shape = (m*N,n*N)
    return out


def bar(S0, S1, mlfm, ifx, lam=1e5):
    Cg = [gp.kernel(mlfm.ttc[:, None])
          for gp in mlfm.latentforces]
    for c in Cg:
        c[np.diag_indices_from(c)] += 1e-5
        Lg = [np.linalg.cholesky(c) for c in Cg]
    invcov = block_diag(*[
        cho_solve((L, True), np.eye(mlfm.dim.N*mlfm.dim.R))
        for L in Lg])

    V, v = mlfm._vecK_aff_rep(beta, ifx)
    S_x_I = kron_A_N(S0, mlfm.dim.N*mlfm.dim.K)
    #S_x_I = np.kron(S0, np.eye(mlfm.dim.N*mlfm.dim.K))    
    invcov += lam*V.T.dot(S_x_I).dot(V)
    cov = np.linalg.inv(invcov)
    premean = S1.T.ravel() - v.dot(S_x_I)
    premean = lam*premean.dot(V)

    return np.linalg.lstsq(invcov, premean, rcond=None)[0]

S0, S1 = _get_kf_statistics(X, kf)
ghat = bar(S0, S1, mlfm, ifx)

fig, ax = plt.subplots()
ax.plot(mlfm.ttc, g[0](mlfm.ttc), 'k-', alpha=0.3)
ax.plot(mlfm.ttc, ghat, '+')

########################################################################
# So far this is of limit practical use, it allows us to recover the
# force when we use the operator :math:`\mathbf{P}` evaluated at the
# true force. The next note in the series will consider extending this
# to an iterative EM setting to discover the force.


plt.show()
