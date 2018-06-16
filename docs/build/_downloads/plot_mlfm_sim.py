# -*- coding: utf-8 -*-
"""
Simulating the MLFM
===================

Simulates the model

   .. math::

      \dot{\mathbf{x}(t)} = \\left(\mathbf{L}_x + \mathbf{L}_y g_1(t) + \mathbf{L}_z g_2(t) \\right)\mathbf{x}(t)

where :math:`g_1(t)` and :math:`g_2(t)` are independent Gaussian processes, and :math:`\mathbf{L}_i` are the
standard basis of the Lie algebra :math:`\mathfrak{so}(3)` of the rotation group :math:`SO(3)`.
"""
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
from pydygp.gaussianprocesses import GaussianProcess
from pydygp.linlatentforcemodels import MLFM

# cart to spherical coord
# utility function
def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r

def main():

    np.random.seed(122)    

    # infinitesimal rotation matrices in so(3)
    Lx = np.array([[0., 0., 0.],
                   [0., 0.,-1.],
                   [0., 1., 0.]])

    Ly = np.array([[ 0., 0., 1.],
                   [ 0., 0., 0.],
                   [-1., 0., 0.]])

    Lz = np.array([[ 0.,-1., 0.],
                   [ 1., 0., 0.],
                   [ 0., 0., 0.]])

    # make the basic mlfm model
    mlfm = MLFM([Lx, Ly, Lz])

    # latent force gaussian processes
    g_gps = [GaussianProcess('sqexp', kpar=[1., 1.])
             for r in range(len(mlfm.struct_mats)-1)]

    # simulate some data 
    T = 4.
    Nd = 7
    
    tt = np.linspace(0., T, Nd)

    # dense set of times for solving the ODE
    tt_d = np.linspace(0., T, 100)

    # set of times for simulating the latent trajectory
    tt_gp_sim = np.linspace(tt[0], tt[-1], 25)

    y, y_dense, g, _ = mlfm.sim([0., 0., 1.],
                                tt,
                                gps=g_gps,
                                tt_dense=tt_d,
                                tt_gp_sim=tt_gp_sim,
                                return_gp=True)    

    # plot colors
    fill_col = '#2096BA'
    linecol = '#DF6E21'
    
    # set up orthographic map projection with
    # perspective of satellite looking down at 30N, 70W.
    map = Basemap(projection='ortho',lat_0=30,lon_0=-70,resolution='l')

    # draw lat/lon grid lines every 30 degrees.
    map.drawmeridians(np.arange(0,360,30))
    map.drawparallels(np.arange(-90,90,30))

    az, el, _ = cart2sph(*y_dense.T)
    az_, el_, _ = cart2sph(*y.T)

    lons = np.rad2deg(az)
    lats = np.rad2deg(el)

    # set up orthographic map projection with
    # perspective of satellite looking down at 30N, 70W.
    map = Basemap(projection='ortho',lat_0=30,lon_0=-70,resolution='l')

    # compute native map projection coordinates of lat/lon grid.
    Px_d, Py_d = map(lons, lats)
    P_x, P_y = map(np.rad2deg(az_), np.rad2deg(el_))

    # contour data over the map.
    map.plot(Px_d, Py_d, color=linecol)
    map.scatter(P_x, P_y, color=linecol)

    # draw the edge of the map projection region (the projection limb)
    map.drawmapboundary(fill_color=fill_col)
    plt.show()

if __name__ == '__main__':
    main()
