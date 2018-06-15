import matplotlib.pyplot as plt
import numpy as np
from pydygp.gaussianprocesses import _GaussianProcess
tt = np.linspace(0., 5., 6)
Y = np.sin(tt)
gp = _GaussianProcess('sqexp_gpr')
gp.fit(tt[:, None], Y)
# Make a dense sample
ttd = np.linspace(0., 5., 100)
# predict the mean and covariance
mpred, cpred = gp.pred(ttd[:, None], return_covar=True)
sd = np.sqrt(np.diag(cpred))

fig, ax = plt.subplots()
ax.fill_between(ttd, mpred + 2*sd, mpred-2*sd, alpha=.2)
ax.plot(ttd, mpred, 'k-.')
ax.plot(tt, Y, 's')
plt.show()
