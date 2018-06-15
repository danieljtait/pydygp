import numpy as np
from scipy.integrate import odeint
from pydygp.linlatentforcemodels import MLFM_MH_AdapGrad as mlfm

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

"""
Model setup 
"""

# 3 basis matrices of the Lie algebra so(2)

Lx = np.array([[0., 0., 0.],
               [0., 0.,-1.],
               [1., 0., 0.]])

Ly = np.array([[0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.]])

Lz = np.array([[0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.]])

mlfm_mod = mlfm([np.zeros((3, 3)), Lx, Ly, Lz])

# Simulate some data
tt = np.linspace(0., 5., 5)
y = np.random.normal(size=tt.size*3).reshape(tt.size, 3)

mlfm_mod.setup(data_times=tt, data_Y=y, aug_times='linspace')

