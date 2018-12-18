""" Base class definition for the Multiplicative Latent Force Model
"""
import numpy as np
from collections import namedtuple
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels as sklearn_kernels

def _make_tt_dense(tt, dt_max):
    # utility function for BaseMLFM.sim(...)
    res, inds = ([tt[0]], [0])
    for ta, tb in zip(tt[:-1], tt[1:]):
        N = int(np.ceil((tb - ta) / dt_max + 1))
        _tt = np.linspace(ta, tb, N)
        res = np.concatenate((res, _tt[1:]))
        inds.append(inds[-1] + N - 1)
    return res, inds

def _sim_gp(tt, gp):
    """Simulates values from a sklearn GPR object
    """
    K = gp.kernel(tt[:, None])
    K[np.diag_indices_from(K)] += gp.alpha
    L = np.linalg.cholesky(K)
    return L.dot(np.random.normal(size=tt.size))

# keep track of the dimensions in the MLFM
Dimensions = namedtuple('Dimensions', 'N K R D')

class BaseMLFM:
    """
    Base class for the Multiplicative Latent Force Model.

    Parameters
    ----------

    basis_mats : tuple
        A tuple of square matrices

    lf_kernels : list, optional
        Kernels of the latent force Gaussian process objects
    
    """
    def __init__(self, basis_mats, R=None, lf_kernels=None):

        self.basis_mats = basis_mats

        if R is None:
            if lf_kernels is None:
                raise ValueError("If R is not supplied then lf_kernels must be supplied.")
            else:
                R = len(lf_kernels)

        # store the model dimensions
        K = basis_mats[0].shape[0]
        D = len(basis_mats)
        self.dim = Dimensions(None, K, R, D)

        # setup the latent forces
        self.setup_latentforces(lf_kernels)

    def setup_latentforces(self, kernels=None):
        """Initalises the latent force GPs

        Parameters
        ----------

        kernels : list, optional
            Kernels of the latent force Gaussian process objects

        """
        if kernels is None:
            # Default is for kernels 1 * exp(-0.5 * (s-t)**2 )
            kernels = [sklearn_kernels.ConstantKernel(1.) *
                       sklearn_kernels.RBF(1.) for r in range(self.dim.R)]

        if len(kernels) != self.dim.R or \
           not all(isinstance(k, sklearn_kernels.Kernel) for k in kernels):
            _msg = "kernels should be a list of {} kernel objects".format(self.dim.R)
            raise ValueError(_msg)

        self.latentforces = [GaussianProcessRegressor(kern) for kern in kernels]

    def sim(self, x0, tt, beta=None, dt_max=0.1, latent_forces=None, size=1):
        """Simulate the process along a set of points

        Parameters
        ----------

        x0 : array_like, shape (K, )
            initial condition for the ode

        tt : array_like
            Ordered sequence of time points for the model to be simulated at

        beta : array_like, shape (R+1, D)

        dt_max : float, defualt = 0.1
            Maximum spacing of dense time points used for simulating the model.

        latent_forces : Tuple of callables, optional, default = None

        Returns
        -------

        Y : array, shape(len(tt), K)
            Values of the MLFM simulated at tt

        latent_forces : tuple of callables
            Tuple of functions (g_1(t),...,g_R(t)) used to simulate the model.


        Examples
        --------
    
        >>> import numpy as np
        >>> from pydygp.linlatentforcemodels import BaseMLFM
        >>> struct_mats = [np.zeros((2, 2))] + [*pydygp.liealgebras.so2()]
        >>> tt = np.linspace(0., 5., 15)
        >>> mlfm = BaseMLFM(struct_mats)
        >>> Y, g = mlfm.sim([1., 0], tt)

        """
        from scipy.interpolate import interp1d
        from scipy.integrate import odeint

        if beta is None:
            try:
                # some of the child classes
                # will fix beta
                beta = self.beta  
            except:
                raise ValueError("Must supply beta.")
        
        if beta.shape[0] != self.dim.R + 1 \
           or beta.shape[1] != self.dim.D:
            msg = "Beta must be of shape {} x {}.".format(self.dim.R+1,
                                                          self.dim.D)
            raise ValueError(msg)
        # create a dense set of time points
        # so that max(np.diff(ttdense)) <= dt_max
        ttdense, inds = _make_tt_dense(tt, dt_max)

        # allows the passing of pre-defined functions
        if latent_forces is None:
            ginterp = [interp1d(ttdense,
                                _sim_gp(ttdense, lf),
                                kind='cubic',
                                fill_value='extrapolate')
                       for lf in self.latentforces]
        else:
            assert(len(latent_forces) == self.dim.R)
            ginterp = latent_forces

        # form the coeff. matrices of the evol. equation from
        # the structural parameters
        struct_mats = [sum(brd*Ld
                           for brd, Ld in zip(br, self.basis_mats))
                       for br in beta]

        # the evolution equation
        def dXdt(X, t):
            At = struct_mats[0] + \
                 sum(Ar*ur(t) for Ar, ur in zip(struct_mats[1:],
                                                ginterp))
            return At.dot(X)

        if size == 1:
            sol = odeint(dXdt, x0, ttdense)            
            return sol[inds, :], ginterp
        else:
            return [odeint(dXdt, x0i, ttdense)[inds, :] for x0i in x0], ginterp


    def _odeint(self, x0, tt, beta, glist):
        if len(glist) != self.dim.R:
            raise ValueError("glist must be a list of {} callable functions.".format(self.dim.R))

        # import odeint for solving
        from scipy.integrate import odeint
        
        struct_mats = [sum(brd*Ld
                           for brd, Ld in zip(br, self.basis_mats))
                       for br in beta]

        def A(t):
            return struct_mats[0] + sum(Ar*gr(t)
                                        for Ar, gr in zip(struct_mats[1:], glist))

        return odeint(lambda x, t: A(t).dot(x), x0, tt)


    def _component_functions(self, g, beta, N=None):
        if N is None:
            N = self.dim.N

        g = g.reshape(self.dim.R, N)
        g = np.row_stack((np.ones(N), g))

        struct_mats = np.array([
            sum(brd*Ld for brd, Ld in zip(br, self.basis_mats))
            for br in beta])

        # match struct mats and G on r=0,...,R
        comp_funcs = struct_mats[..., None] * g[:, None, None, :]

        # sum over r
        return comp_funcs.sum(0)

    def __mul__(self, other):
        return MLFMCartesianProduct(self, other)

    def flatten(self):
        return self,
    

class MLFMCartesianProduct:
    """
    Cartesian product of two MLFM like objects
    """
    def __init__(self, mlfm1, mlfm2):
        # Check compatability
        N1, K1, R1, D1 = mlfm1.dim
        N2, K2, R2, D2 = mlfm2.dim

        self.mlfm1 = mlfm1
        self.mlfm2 = mlfm2
        
        if R1 != R2:
            raise ValueError("The number of latent forces in both ",
                             "both models must be the same")

        if N1 != N2:
            raise ValueError("Dimension of the input vector must ",
                             "be the same for both models")

        # This causes a bit of rewrite
        self.dim = Dimensions(N1, None, R1, None)
        
    def __mul__(self, mlfm):
        return MLFMCartesianProduct(self, mlfm)

    def flatten(self):
        """
        Returns a flat tuple of the constituents of the MMLF objects
        """
        mlfms = self.mlfm1.flatten() + self.mlfm2.flatten()
        return mlfms

        
    def sim(self, x0, tt, beta, dt_max=0.1, size=1, latent_forces=None):
        mlfms = self.flatten()

        # simulate from the first model
        Y0, lf = mlfms[0].sim(x0[0], tt, beta[0], dt_max=dt_max, size=size)

        Data = [Y0, ]
        for i in range(len(mlfms)-1):
            yi, _ = mlfms[i+1].sim(x0[i+1], tt,
                                   latent_forces=lf,
                                   beta=beta[i+1], dt_max=dt_max, size=size)
            Data.append(yi)
        return Data, lf
