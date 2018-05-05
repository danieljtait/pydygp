# External imports
from collections import namedtuple
import numpy as np

# Package imports
from pydygp.gaussianprocesses import GaussianProcess
from pydygp.kernels import Kernel

Dimensions = namedtuple('Dimensions', 'N K R')

# Container object for the model parameter set 
MHParameters = namedtuple('MHParameters',
                          'xcur gcur')

class MLFM_MH_AdapGrad:
    def __init__(self,
                 structure_matrices):

        self.struct_mats = np.asarray(structure_matrices)
        self.dim = Dimensions(N=None,
                              K=self.struct_mats.shape[1],
                              R=self.struct_mats.shape[0]-1)


        """
        Initalise the container for mhparameters
        """
        self.mh_pars = MHParameters(xcur=[(None, None) for k in range(self.dim.K)],
                                    gcur=[(None, None) for r in range(self.dim.R)])

        """
        various flags for model customisation
        """
        # If true then user supplied obs probability
        # else iid diagonal Gaussians
        self.pobs_is_custom = False

        # default behaviour is a Gibbs update
        self.x_update_type = 'gibbs'

        # True if additional latent variables not corresponding
        # to the data times are to be simulated
        self.aug_latent_vars = False

    """
    Allow the specification of customised distributions for
    the observation error distribution

    pobs should be of the form

        pobs(y, x, par)  | returns probability of y given x and par
    
    """
    def obs_dist_setup(self,
                       pobs=None,
                       xproposal=None,
                       pobs_par_prior=None,
                       pobs_par_proposal=None):

        if pobs is not None:
            try:
                assert(xproposal is not None)
            except:
                raise ValueError("If specifying a custom observation distribution the user must also supply custom 'xproposal' method")

            self.pobs = pobs

            # flag the new proposal distribution for x and
            # that updating is now done using a mh scheme
            self._xproposal = xproposal
            self.x_update_type = 'mh'

            # methods relating to the updating of the parameter govenrning
            # the observaion distribution
            self.pobs_par_prior = pobs_par_prior
            self.pobs_par_proposal = pobs_par_proposal

    """
    specification of the gaussian process interpolators for the
    latent trajectory
    """
    def x_gp_setup(self,
                   kern_type='sqexp',
                   kern=None,
                   kpar=None):
        x_gp_setup(self, kern_type, kern, kpar)


    """
    specification of the latent gaussian forces
    """
    def g_gp_setup(self,
                   kern_type='sqexp',
                   kern=None,
                   kpar=None):
        g_gp_setup(self, kern_type, kern, kpar)

    # Customisation method for introducing an additional
    # set of latent states to be observed 
    def add_latent_states(self, aug_t):
        self.aug_latent_vars = True
        sort_augmented_timeset(obj, aug_t)


    """
    latent variable initalisation
    """
    def init_latent_vars(self,
                         x0s=None,
                         g0s='prior'):
        init_latent_vars(self, x0s, g0s)


    """
    set data and times
    """
    def fit(self, data_times, data_Y=None, aug_t=None):

        # attach the data time
        self.data_times = data_times

        if aug_t is not None:
            self.add_latent_states(aug_t)


        self.init_latent_vars()


    """
    X updating
    """
    def xupdate(self):
        if self.x_update_type == 'gibbs':
            self._x_gibbs_update()

        elif self.x_mh_update == 'mh':
            self._x_mh_update()


    # gibbs update of the 
    def _x_gibbs_update(self):
        _x_gibbs_update(self)
    
    """
    G updating
    """
    def gupdate(self):
        _g_gibbs_update(self)



"""
Methods related to model setup
"""
def x_gp_setup(obj, kern_type, kern, kpar):
    if kpar is None:
        kpar = [None for k in range(obj.dim.K)]
    
    if kern_type == 'sqexp':
        x_kernels = [GradientMultioutputKernel.SquareExponKernel(kp)
                     for kp in kpar]


def g_gp_setup(obj, kern_type, kern, kpar):
    if kpar is None:
        kpar = [None for k in range(obj.dim.R)]    

    if kern_type == 'sqexp':
        g_kernels = [Kernel.SquareExponKernel(kp)
                     for kp in kpar]
    obj.g_gps = [GaussianProcess(kern) for kern in g_kernels]


##
# creates a sorted full time array from the data
# and augmented times while keeping track of indices
# in the full timeset that correspond to a data point
def sort_augmented_timeset(obj, aug_t):
    data_t = obj.data_times
    

"""
Methods related to model initialisation
"""
def init_latent_vars(obj, x0, g0s):

    """
    Trajectory initalisation
    """
    # User supplied initial value
    if isinstance(x0, np.ndarray):
        obj.mh_pars.xcur = (x0, None)

    # Set inital latent variables equal to state
    elif x0 == 'data':
        assert(not obj.aug_latent_vars)
        obj.mj_pars.xcur = (obj.data_Y.copy(), None)

    # optional fit gp to observed data and then use the fitted
    # process to estimate latent states
    elif x0 == 'gpfit':
        pass 


    """
    Latent force initalisation
    """

    # User supplied inital force values
    if isinstance(g0s, list):
        for r in range(obj.dim.R):
            obj.mh_pars.gcur[r] = (g0s[r], None)

    elif g0s == 'prior':
        tt = obj.data_times[:, None]
        for r, gp in enumerate(obj.g_gps):
            gp.fit(tt)
            rv = gp.sim()
            obj.mh_pars.gcur[r] = (rv, None)

"""
Methods for carrying out Monte-Carlo updates
of the latent trajectories
"""
def _x_gibbs_update(obj):
    pass

def _x_mh_update(obj):
    pass

"""
Methods for carry out Monte-Carlo updates
of the latent forces
"""
def _g_gibbs_update(obj):
    pass
