from pydygp.kernels import Kernel, GradientKernel

# Default settings

X_KERN_DEFAULT = 'sqexp'
GAMMAS_DEFAULT = 0.


class MLFM_AdapGrad:
    """
    The underlying class for the Multiplicative Latent Force Model
    where the model fitting is done using the Adaptive Gradient Matching
    approach
    """
    def __init__(self, structure_matrices):
        pass


    def setup(self, data_times, data_Y=None, aug_times=None, **kwargs):
        """
        default setup function carries out model initalisation
        """
        # Attachement of data and times
        self.data_times = data_times
        self.data_Y = data_Y

        # Update dimensions to add number of data points
        self.dim = Dimensions(data_times.size, self.dim.K, self.dim.R)

        # setup of the model variables called in a way respecting
        # the model hierarchy
        self._gammas_setup()
        self._xp_setup(aug_times, **kwargs)

    def _gammas_setup(self, gammas=None):
        if gammas == None:
            self.gammas = GAMMAS_DEFAULT*np.ones(self.dim.K)

        else:
            self.gammas = gammas

    def _x_gp_setup(self, aug_times=None, x_kern='sqexp', x_kpar=None, **kwargs):
        """ Sets up the latent gp kernels
        """

        # enhances the model with additional latent input times
        if aug_times is not None:
            add_latent_states(self, self.data_times, aug_times, **kwargs)

    def x_cond_posterior(self, x, Y=None, gs=None, x_kpars=None):
        """ Evaluates the conditional posterior

        .. math::
           p(\mathbf{x}|\mathbf{y}, \mathbf{g}, \\boldsymbol{\phi})

        [more explanation]
        """
        if Y is None:
            Y = self.data_Y

        if x_kpars is None:
            x_kpars = [None for k in range(self.dim.K)]

        fks = None

        logp = 0.

        for k, gp in enumerate(self.x_gps):
            
            Mk, Sk_chol, Cxx_chol = dx_gp_condmats(tt, kern, x_kpars[k], 'chol')

            
