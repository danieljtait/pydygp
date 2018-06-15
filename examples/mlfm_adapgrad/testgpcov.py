import numpy as np
from pydygp.kernels import GradientKernel
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

kse = GradientKernel.SquareExponKernel()

tt = np.linspace(0., 2, 3)

Cxx = kse.cov(tt[:, None])
Cxdx = kse.cov(tt[:, None], comp='xdx')
Cdxdx = kse.cov(tt[:, None], comp='dxdx')
print(Cxx)
print(Cxdx)
print(Cdxdx)
