import numpy as np
import matplotlib.pyplot as plt
from pydygp.gaussianprocesses import GaussianProcess
from pydygp.linlatentforcemodels import MLFM
# make the infinitesimal rotation matrices
np.random.seed(125)
Lx = np.array([[0., 0., 0.], [0., 0.,-1.], [0., 1., 0.]])
Ly = np.array([[0., 0.,-1.], [0., 0., 0.], [1., 0., 0.]])
Lz = np.array([[0.,-1., 0.], [1., 0., 0.], [0., 0., 0.]])
gp = GaussianProcess("sqexp", kpar=[.1, 1.])
gp.jitter = False
mlfm = MLFM([Lz, Lx+Ly])
tt = np.linspace(0., 5., 10)
X, gval, ttd, xxd = mlfm.sim([0., 0., 1.], tt, gps=(gp, ), return_gp=True)
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.plot(tt, X)
ax2 = fig.add_subplot(122)
ax2.plot(ttd, gval, '+')
plt.show()
