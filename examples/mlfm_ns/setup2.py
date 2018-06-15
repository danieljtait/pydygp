import numpy as np
from pydygp.linlatentforcemodels import MLFM_MH_NS

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

Lx = np.array([[0., 0., 0.],
               [0., 0., 1.],
               [0.,-1., 0.]])

mlfm = MLFM_MH_NS([np.zeros((3, 3)), Lx])

tt = np.linspace(0., 3., 5)

mlfm.time_interval_setup(tt, h=0.2)

mlfm.phi_setup()
mlfm.phi_init(phi_val=[(1., 1.)]*3)

mlfm.x_gp_setup()

mlfm.x_cond_meanvar()


