import numpy as np

class BaseKernel:
    """
    base class for a parameterised kernel function
    """
    def __init__(self, cov_method, kpar, par_grad=None):
        self.cov_method = cov_method
        self.kpar = kpar
        self.par_grad = par_grad

    def cov(self, x1, kpar=None, x2=None):
        if kpar is None:
            kpar = self.kpar
        
        if not isinstance(x1, np.ndarray):
            x1 = np.asarray(x1)

        if not isinstance(x2, (float, list, np.ndarray)):
            if x2 is None:
                x2 = x1.copy()
            else:
                raise ValueError("Unrecognised input 'x2' to kernel covariance function")

        return self.cov_method(x1, x2, kpar)

    def cov_par_grad(self, kpar, x1, x2=None, ind=-1):

        if not isinstance(x1, np.ndarray):
            x1 = np.asarray(x1)

        if not isinstance(x2, (float, list, np.ndarray)):
            if x2 is None:
                x2 = x1.copy()
            else:
                raise ValueError("Unrecognised input to kernel covariance function")

        try:
            return self.par_grad(x1, x2, kpar, ind)
        except:
            pass


class SquareExponentialKernel(BaseKernel):

    def __init__(self, dim=1, kpar=None):

        def cov_method(xx1, xx2, par):
            xs = [np.meshgrid(x2, x1) for x1, x2 in zip(xx1.T, xx2.T)]
            exp_arg = sum(p*(x[0]-x[1])**2 for (x, p) in zip(xs, par[1:]))
            return par[0]*np.exp(-exp_arg)

        def par_grad(xx1, xx2, par, ind=-1):
            if ind < 0:
                # return the gradient with respect to all of the parameters
                cov = cov_method(xx1, xx2, par)
                
                dkdp0 = cov_method(xx1, xx2, np.concatenate(([1.], par[1:])))
                xs = [np.meshgrid(x2, x1) for x1, x2 in zip(xx1.T, xx2.T)]
                dkdpi = [-(x[0]-x[1])**2*cov for x in xs]
                return [dkdp0]+dkdpi

            elif ind == 0:
                return cov_method(xx1, xx2, np.concatenate(([1.], par[1:])))

            else:
                cov = cov_method(xx1, xx2, par)
                x2, x1 = np.meshgrid(xx2[:, ind-1], xx1[:, ind-1])
                return -(x2-x1)**2*cov

        if kpar is None:
            kpar = np.ones(dim+1)
        
        super(SquareExponentialKernel, self).__init__(cov_method,
                                                      kpar,
                                                      par_grad=par_grad)


class Kernel:

    def __new__(cls, ktype, *args, **kwargs):
        return Kernel.factory(ktype)

    @staticmethod
    def factory(type):
        if type == "sqexp":
            return SquareExponentialKernel()

        else:
            raise TypeError("Unrecognised type '{}' kernel.".format(str(type)))


"""
***********************************
Gaussian Process updating functions
***********************************
"""

class BaseGaussianProcess:
    def __init__(self, kernel):
        try:
            assert(isinstance(kernel, BaseKernel))
        except:
            raise ValueError("'kernel' must be a valid kernel object")


class BayesGaussianProcess(BaseGaussianProcess):
    def __init__(self, kern, priors=None):
        self.kpar_priors = priors
        super(BayesGaussianProcess, self).__init__(kern)


class HMCGaussianProcess(BaseGaussianProcess):
    """
    Allows Gaussian Process hyper-parameter inference
    using the Hybrid Monte Carlo method 
    """

def gp_scale_update(gpobj):
    if isinstance(gpobj.kernel, SquareExponentialKernel):
        pass

class GaussianProcessManager:
    """
    Handles the creation of Gaussian Process objects
    """

    @staticmethod
    def bayes_gp(ktype='sqexp', prior='ind_gamma', **kwargs):
        if ktype == 'sqexp':
            # Creating a Bayesian Gaussian Process with Square Exponential Kernel
            try:
                dim = kwargs['dim']
            except:
                dim = 1
            kern = Kernel('sqexp')
            priors = [Gamma(**kwargs) for d in range(dim+1)]
            return BayesGaussianProcess(kern, priors)

    @staticmethod
    def hmc_bayes_gp(ktype='sqexp', prior='ind_gamma', proposal='ind_rw', **kwargs):
        pass

    @staticmethod
    def mcmc_bayes_gp(ktype='sqexp', prior='ind_gamma', proposal='ind_rw', **kwargs):
        pass

class Gamma:
    def __init__(self):
        pass
        
k = Kernel('sqexp')

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
tt = np.linspace(0., 1., 3)

gp = GaussianProcessManager.bayes_gp()
print(gp)
