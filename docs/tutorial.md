# Tutorial

## Complete Gaussian Process Example

This is a simple example to demonstrate the creation of a simple GaussianProcess object from a Kernel object and then demonstrate the fit and prediction intervals for a simple sin curve.

```python
from pydygp.kernels import Kernel
from pydygp.gaussianprocesses import GaussianProcess

# Additional imports
import numpy as np
import matplotlib.pyplot as plt

# Initialises a default square exponential kernel: k(s, t) = exp(-(s-t)**2)
kse1 = Kernel.SquareExponKernel()

# initalise a square exponential kernel with given parameters
kpar = [.25, 2.5]
kse2 = Kernel.SquareExponKernel(kpar)

# create a Gaussian processes from the kernels
# and then fit it to the simulated data
gp1 = GaussianProcess(kse1)
gp2 = GaussianProcess(kse2)

# create some data
xx = np.linspace(0, 2*np.pi, 9)
Y = np.sin(xx)

# fit the Gaussian processes
for gp in [gp1, gp2]:
    gp.fit(xx[:, None], y=Y)

# predict values at new points and plot results 
xnew = np.linspace(-1, 7, 150)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title(r"GP prediction with kernel $k(s, t) = \theta_1e^{-\theta_2(s-t)^2}$")
for gp in [gp1, gp2]:
    pred_mean, pred_cov = gp.pred(xnew[:, None], return_var=True)
    sd = np.sqrt(np.diag(pred_cov))
    ax.fill_between(xnew, pred_mean + 2*sd, pred_mean - 2*sd, alpha=0.5)
    lab = r"$\theta$ = ({}, {})".format(*gp.kernel.kpar)
    ax.plot(xnew, pred_mean, label=lab)
    ax.legend(loc='upper right')

ax.plot(xx, Y, 'ks')
plt.show()
```