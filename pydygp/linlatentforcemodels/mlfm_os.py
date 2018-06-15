import numpy as np
from pydygp import nssolve

from collections import namedtuple
from scipy.stats import invgamma, gamma, multivariate_normal, wishart
from scipy.linalg import block_diag

from pydygp.gaussianprocesses import (GaussianProcess,
                                     _GaussianProcess)

from .basemlfm import BaseMLFM

Dimensions = namedtuple('Dimensions', 'N K R')

# Default options for setup
default_t_setup = {'h': None }
default_op_setup = {'ifix': 0}

# Model variable names
var_names = ('phi', 'psi', 'x_gp')

class MLFM_NS_SS(BaseMLFM):
    def __init__(self, struct_mats):
        super(MLFM_NS_SS, self).__init__(struct_mats)

        # Structure matrices
        self.dim = Dimensions(N=None,
                              K=self.struct_mats[0].shape[0],
                              R=len(struct_mats)-1)

        # order is 1 for the single-step fixed point method
        self.order = 1

        # Flags for variable setup
        for vn in var_names:
            setattr(self, "{}_is_setup".format(vn), False)

        # Flags for variable initalisation
        for vn in var_names:
            setattr(self, "{}_is_init".format(vn), False)

        self.is_verbose = False


    def setup(self, data_times, **kwargs):
        """
        Handles the default set of model variables and their priors
        """

        try:
            self.time_interval_setup(data_times, **kwargs['t_setup'])
        except:
            self.time_interval_setup(data_times, **default_t_setup)

        try:
            self.operator_setup(**kwargs['op_setup'])
        except:
            self.operator_setup(**default_op_setup)


    def time_interval_setup(self, data_times, h=None):
        """
        Sets up the intervals and handles the number of time points
        in each intervals

        Completes the time set and so requires the times for which there
        are data observations to be specified.
        """
        intervals = [nssolve.ns_util.Interval(ta, tb)
                     for ta, tb in zip(data_times[:-1], data_times[1:])]
        data_inds = [0]
        for I in intervals:
            I.set_quad_style(h=h)
            data_inds.append(data_inds[-1]+I.tt.size-1)
        
        self.intervals = intervals
        self.comp_times = nssolve.ns_util.get_times(self.intervals)
        self.data_inds = data_inds
        self.dim = Dimensions(self.comp_times.size, self.dim.K, self.dim.R)


    def operator_setup(self, ifix=0):
        self.t0 = self.intervals[ifix].ta
        nsop = nssolve.QuadOperator(fp_ind=ifix,
                                    method='single_fp',
                                    intervals=self.intervals,
                                    K=self.dim.K,
                                    R=self.dim.R,
                                    struct_mats=self.struct_mats,
                                    is_x_vec=True)
        self.nsop = nsop

    def g_gp_setup(self, gps=None, kernels=None, ktypes=None):

        if gps is not None:
            assert(len(gps) == self.dim.R)
            self.g_gps = gps

        # Kernels should be a tup of length R with valid
        # kernel objects
        elif kernels is not None:
            assert(len(kernels) == self.dim.R)

        elif ktypes is not None:
            assert(len(ktypes) == self.dim.R)
            
