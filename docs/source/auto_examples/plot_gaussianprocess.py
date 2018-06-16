# -*- coding: utf-8 -*-
"""
Simulate and Fit a Gaussian Process
===================================

.. currentmodule:: pydygp.gaussianprocesses

A simple example of initalising a :class:`GaussianProcess`, simulating some observations
and then returning the predicted mean and covariance.

"""

import numpy as np
from pydygp.gaussianprocesses import GaussianProcess
import matplotlib.pyplot as plt

np.random.seed(15)

def main():
    x = np.linspace(0., 2*np.pi, 7)

    # Initalise a GP with the square exponential kernel
    gp = GaussianProcess('sqexp')

    # simulate some observations
    y = gp.sim(x[:, None])

    gp.fit(x[:, None], y)

    # Dense sample of x
    xd = np.linspace(x[0], x[-1], 100)

    mean, cov = gp.pred(xd[:, None], return_covar=True)
    sd = np.sqrt(np.diag(cov))

    plt.figure()
    plt.fill_between(xd, mean - 2*sd, mean + 2*sd, alpha=0.2)
    plt.plot(xd, mean, '-r')
    plt.plot(x, y, 'ko')
    plt.show()

if __name__ == '__main__':
    main()

