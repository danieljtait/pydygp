import numpy as np
from pydygp.kernels import Kernel

i = 0
eps = 1e-6
p0 = np.array([1.1, 1.2])
pp = p0.copy()
pp[i] += eps

kse = Kernel.SquareExponKernel(kpar=p0)
ksep = Kernel.SquareExponKernel(kpar=pp)

tt = np.linspace(0., 1., 3)

cov = kse.cov(tt[:, None])
covp = kse.cov(tt[:, None], kpar=pp)

print((covp-cov)/eps)

dCdp = kse.cov_par_grad(kse.kpar, tt[:, None])
print(dCdp[0])


