import numpy as np
import matplotlib.pyplot as plt
from pydygp.gaussianprocesses import GaussianProcess

np.random.seed(11)

gp = GaussianProcess('sqexp', kpar=[1.5, 1.1])

x = np.linspace(0., 5., 10)
y = gp.sim(x[:, None])

# call to gp fit - creates covar. matrix
gp.fit(x[:, None])

par_hat, msg = gp.hyperpar_optim(y)
print(par_hat, msg)
