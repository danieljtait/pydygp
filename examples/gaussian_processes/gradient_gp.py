import numpy as np
from pydygp.kernels import Kernel, GradientKernel
from pydygp.gaussianprocesses import GradientGaussianProcess

kpar = np.array([1., .2])
k = Kernel.SquareExponKernel(kpar)
kse = GradientKernel.SquareExponKernel(kpar, dim=1)


xx = np.linspace(0., 5., 3)



gp = GradientGaussianProcess(kse)
gp.fit(x=xx[:, None],
       grad_x=xx[:, None])


