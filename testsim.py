import numpy as np
import matplotlib.pyplot as plt
from pydygp.linlatentforcemodels import MLFM
from pydygp.gaussianprocesses import GaussianProcess
from mpl_toolkits.basemap import Basemap

np.random.seed(122)

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
g_gps = [GaussianProcess('sqexp', kpar=[1., 1.]) for r in range(len(mlfm.struct_mats)-1)]

T = 4.

# simulate some data
tt = np.linspace(0., T, 7)

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


# utility function
def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r

fig = plt.figure()
ax1 = plt.subplot2grid((1, 5), (0, 0), colspan=2)

# set up orthographic map projection with
# perspective of satellite looking down at 30N, 70W.
map = Basemap(projection='ortho',lat_0=30,lon_0=-70,resolution='l')

# draw the edge of the map projection region (the projection limb)
map.drawmapboundary(fill_color='aqua')
# draw lat/lon grid lines every 30 degrees.
map.drawmeridians(np.arange(0,360,30))
map.drawparallels(np.arange(-90,90,30))

linecol = '#D73A31'
az, el, _ = cart2sph(*y_dense.T)
az_, el_, _ = cart2sph(*y.T)

lons = np.rad2deg(az)
lats = np.rad2deg(el)

# compute native map projection coordinates of lat/lon grid.
x_d, y_d = map(lons, lats)
_x, _y = map(np.rad2deg(az_), np.rad2deg(el_))

# contour data over the map.
map.plot(x_d, y_d, color=linecol)
map.scatter(_x, _y, color=linecol)

ax2 = plt.subplot2grid((1, 5), (0, 2), colspan=3)
ax2.plot(tt_d, y_dense, 'k-', alpha=0.2)
ax2.plot(tt, y, 'ks')
ax2.spines['left'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.yaxis.set_ticks_position('right')
ax2.set_xlabel('Time')
#cs = map.contour(x,y,wave+mean,15,linewidths=1.5)
#plt.title('contour lines over filled continent background')

#fig, ax = plt.subplots()
#ax.plot(tt_d, y_dense, 'k-', alpha=0.2)
#ax.plot(tt, y, 's')
#ax.plot(tt_d, g, 'k-', alpha=0.2)
plt.tight_layout()
plt.show()

