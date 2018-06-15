import numpy as np
import matplotlib.pyplot as plt
#from pydygp.gaussianprocesses import _GaussianProcess
from pydygp.gaussianprocesses import GaussianProcess

tt = np.linspace(0., 5., 6)
Y = np.sin(tt)

gp = GaussianProcess('sqexp', kpar=[1., 3.])

fig, ax = plt.subplots()

# densly sample the interval to display predictions
ttd = np.linspace(tt[0]-1., tt[-1]+1., 100)

for par, col in zip([.5, 1., 2], ['b', 'r', 'g']):

    gp.fit(tt[:, None], y=Y, kpar=[1., par])
    mpred, cpred = gp.pred(ttd[:, None], return_covar=True)

    var = np.diag(cpred)
    sd = np.sqrt(var)
    
    ax.fill_between(ttd, mpred + 2*sd, mpred - 2*sd,
                    alpha=0.2, facecolor=col)
    ax.plot(ttd, mpred, col + '-')

# plot the true values
ax.plot(tt, Y, 's')



plt.show()
