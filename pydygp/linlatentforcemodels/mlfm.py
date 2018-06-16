import numpy as np
from .mlfm_os import MLFM_NS_SS

class MLFMFactory:
    """
    Handles the creation of mlfm models
    """

    @staticmethod
    def onestep(struct_mats):
        return MLFM_NS_SS(struct_mats)


class MLFM:

    def __init__(self, struct_mats):
        self.struct_mats = np.asarray(struct_mats)

    def vec_flow(self, x_list, g_list):
        """
        Arguments should be list types where each entry is the
        kth dimension of X and rth latent force respectively
        """
        pass
        # column stack the variables
        #X = column_stack(x_list)        
        #G = column_stack(g_list)
        #
        #gAs = [struct_mats[0] + sum([gt_k*struct_mats[1:]) for gt_k in grow)
        #       for grow in G]

        # F(t, x) = [A0 + sum(g(t)*Ar)]x(t)
        #F = np.array([np.dot(At, x) for At, x in zip(gAs, X)])
        #return F


    def sim(self, x0, tt, gs=None, gps=None, return_gp=False):
        """
        Simulates a realisation of the mlfm using either a supplied
        collection of forces or else by simulating from a provided set
        of latent Gaussian processes

        .. note::

           None of the attributes of the :class:`MLFM` are altered and only
           the model structure matrices provided to :classmethod:`MLFM.__init__`
           are required
        """
        from scipy.integrate import odeint

        struct_mats = self.struct_mats

        if gs is None and gps is None:
            raise ValueError("Either a function or Gaussian Process is required for simulating")
        
        # dimensional rationality checks
        if gs is not None:
            assert(isinstance(gs, (tuple, list)))
            assert(len(gs) == len(struct_mats)-1)

        if gs is None and gps is not None:
            gs, gval, ttd, data_inds = _sim_gp_interpolators(tt, gps, return_gp)
        
        def dXdt(X, t):
            # constant part of flow
            A0 = struct_mats[0]

            # time dependent part
            At = sum([Ar*gr(t) for Ar, gr in zip(struct_mats[1:], gs)])

            return np.dot(A0 + At, X)

        sol = odeint(dXdt, x0, tt)
        print(len(tt), len(data_inds))
        if return_gp:
            # dense solution
            sold = odeint(dXdt, x0, ttd)
            sol = sold[data_inds, :]
            return sol, gval, ttd, sold
        else:
            return sol

    @staticmethod
    def ns(struct_mats, order=1):
        return MLFM_NS(struct_mats, order=1)


class MLFM_NS(MLFM):

    def __init__(self, struct_mats, order=1):
        super(MLFM_NS, self).__init__(struct_mats)
        self.order = order


class MLFM_MH_NS(MLFM):
    def __init__(self, structure_matrices):
        super(MLFM_MH_AdapGrad, self).__init__(structure_matrices)


def _sim_gp_interpolators(tt, gps, return_gp):
    """
    simulates values from the latent Gaussian process objects
    over a dense grid and returns the a collection of numpy interpolating
    functions
    """
    from scipy import interpolate

    data_inds = [0]
    ttdense = []
    for ta, tb in zip(tt[:-1], tt[1:]):
        ttdense = np.concatenate((ttdense, np.linspace(ta, tb, 5)[:-1]))
        data_inds.append(ttdense.size)

    ttdense = np.concatenate((ttdense, [tt[-1]]))


    # simulate at the sparse values
    rvs = [gp.sim(tt[:, None]) for gp in gps]

    # predict at the dense values
    for rv, gp in zip(rvs, gps):
        gp.fit(tt[:, None], rv)
    rvs = [gp.pred(ttdense[:, None]) for gp in gps]

    # return the linear interpolator
    gs = [interpolate.interp1d(ttdense, rv, fill_value='extrapolate') for rv in rvs]

    if return_gp:
        return gs, np.column_stack(rvs), ttdense, data_inds
    else:
        return gs, None, None, None
