import numpy as np

class BaseMLFM:

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
        
        # dimensional rationality checks
        if gs is not None:
            assert(isinstance(gs, (tuple, list)))
            assert(len(gs) == len(struct_mats)-1)

        if gs is None and gps is not None:
            gs, gval, ttd = _sim_gp_interpolators(tt, gps, return_gp)

        def dXdt(X, t):
            # constant part of flow
            A0 = struct_mats[0]

            # time dependent part
            At = sum([Ar*gr(t) for Ar, gr in zip(struct_mats[1:], gs)])

            return np.dot(A0 + At, X)


        sol = odeint(dXdt, x0, tt)

        if return_gp:
            return sol, gval, ttd
        else:
            return sol


def _sim_gp_interpolators(tt, gps, return_gp):
    """
    simulates values from the latent Gaussian process objects
    over a dense grid and returns the a collection of numpy interpolating
    functions
    """
    from scipy import interpolate
    
    ttdense = np.concatenate([[ta, 0.5*(ta+tb)]
                              for ta, tb in zip(tt[:-1], tt[1:])])
    ttdense = np.concatenate((ttdense, [tt[-1]]))

    rvs = [gp.sim(ttdense[:, None]) for gp in gps]

    gs = [interpolate.interp1d(ttdense, rv, fill_value='extrapolate') for rv in rvs]

    if return_gp:
        return gs, np.column_stack(rvs), ttdense
    else:
        return gs, None, None