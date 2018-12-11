import matplotlib.pyplot as plt
from scipy.stats import gamma
import numpy as np
from pydygp.probabilitydistributions import InverseGamma
p = InverseGamma(a=2)
q = p.logtransform()
z = gamma.rvs(a=2, size=1000)
x = np.linspace(-5., 5., 100)
plt.plot(x, np.exp(q.logpdf(x)))
plt.hist(-np.log(z), density=True, alpha=0.5)
plt.show()