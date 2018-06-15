import numpy as np
from scipy.stats import multivariate_normal
from pydygp.kernels import Kernel, _Kernel, GradientKernel
import pydygp.kernels
from scipy.optimize import minimize

# Allows kernels to be initalised by strings
univar_kern_names = {'sqexp': lambda : Kernel.SquareExponKernel()}

def kernel_handler(ktype, *args, **kwargs):
    if ktype == 'sqexp':
        return Kernel.SquareExponKernel(*args, **kwargs)

    else:
        raise ValueError
        

class GaussianProcess:
    """
    The basic Gaussian Process class
    """
    def __init__(self, kernel, kpar=None):
        if isinstance(kernel, Kernel):
            self.kernel = kernel

        elif isinstance(kernel, str):
            try:
                self.kernel = kernel_handler(kernel, kpar=kpar)
            except ValueError:
                msg = "'{}' is not a recognised kernel name.".format(kernel)
                raise ValueError(msg)
        else:
            ve_msg = "kernel argument must be Kernel object or valid kernel name"
            raise ValueError(ve_msg)

        self.is_mean_zero = True

        # add small copies of the identity matrix
        # to all covariance matrices that will be inverted
        self.jitter = True
        self.jitter_val = 1e-5


    def fit(self, x, y=None, kpar=None, store_inv_cov=False):
        """
        
        Computes and stores the covariance matrix at input points x

        Parameters
        ----------
        x : function
            The function to be interpolated. It must be a function of a single
            variable of the form ``f(x, a, b, c...)``, where ``a, b, c...`` are
            extra arguments passed in the `args` parameter.
        y : int
            Degree of the interpolating polynomial.
        kpar : {None, [beg, end]}, optional
            Domain over which `func` is interpolated. The default is None, in
            which case the domain is [-1, 1].
        store_inv_cov : tuple, optional
            Extra arguments to be used in the function call. Default is no
            extra arguments.

        """

        if kpar is None:
            kpar = self.kernel.kpar
        else:
            self.kernel.kpar = kpar
        
        self.inputs = x
        self.outputs = y

        # note that kernel.kpar has already been updated
        cov = self.kernel.cov(x)

        if self.jitter:
            cov += np.diag(self.jitter_val*np.ones(cov.shape[0]))

        # compute the Cholesky factor of cov
        L = np.linalg.cholesky(cov)
        self.cov_chol = L
        
        if store_inv_cov:
            self.inv_cov = np.linalg.inv(self.cov)


    def sim(self, x=None, kpar=None):
        """
        Simulates the Gaussian process at the inputs x.
        """
        if x is None:
            x = self.inputs

        if kpar is None:
            kpar = self.kernel.kpar

        cov = self.kernel.cov(x, kpar=kpar)

        # add small copy of I to cov. matrix
        if self.jitter:
            cov += np.diag(self.jitter_val*np.ones(cov.shape[0]))        

        cov_chol = np.linalg.cholesky(cov)
            
        if self.is_mean_zero:
            mean = np.zeros(cov.shape[0])

        rv = np.random.normal(size=x.shape[0])
        rv = np.dot(cov_chol, rv) + mean
        
        return rv

    def pred(self, xnew, return_covar=False):
        """
        Carries out prediction on a fitted Gaussian process.
        """

        # This would allow prediciton of the process at a single
        # point but this is inefficient and probably to be discouraged.
        #if isinstance(xnew, float):
        #    xnew = np.array([xnew]).reshape(1,1)

        c22_chol = self.cov_chol

        C12 = self.kernel.cov(xnew,
                              x2=self.inputs)

        # conditional mean
        mcond = np.dot(C12, back_sub(c22_chol, self.outputs))

        if not return_covar:
            # all we need to do so return conditional mean
            return mcond

        else:
            C11 = self.kernel.cov(xnew)
            ccond = C11 - np.dot(C12, back_sub(c22_chol, C12.T))
            return mcond, ccond


    def loglikelihood(self, y, x=None, kpar=None,
                      return_cov=False, jitter=False):
        """
        Returns the Gaussian process log-likelihood for a finite sample

        """
        if self.is_mean_zero:
            mean = np.zeros(y.size)

        cov = None
        if kpar is not None and x is None:
            cov = self.kernel.cov(self.inputs, kpar=kpar)

        elif kpar is not None and x is not None:
            cov = self.kernel.cov(x, kpar=kpar)

        if cov is None:
            cov = np.dot(self.cov_chol, self.cov_chol.T)

        else:
            if jitter:
                cov += np.diag(self.jitter_val*np.ones(cov.shape[0]))

        try:
            lp = multivariate_normal.logpdf(y, mean=mean, cov=cov)
        except:
            lp = -np.inf

        # Optionally return the calculated covariance
        # for more efficient updating
        if return_cov:
            return lp, cov
        else:
            return lp


    def hyperpar_optim(self, y, p0=None, jitter=False):
        """
        Performs the maximum likelihood estimate of the kernel hyperparameters
        """
        if p0 is None:
            p0 = self.kernel.kpar

        def _objfunc(kpar):
            try:
                return -self.loglikelihood(y, kpar=kpar, jitter=jitter)
            except:
                return np.inf

        def _objfunc_grad(kpar):

            dCdp = self.kernel.cov_par_grad(kpar, self.inputs)

            if self.is_mean_zero:
                m = np.zeros(y.size)
                
            cov = self.kernel.cov(self.inputs, kpar=kpar)
            dOdC = -_mvt_loglik_grad(y, m, cov)

            return [np.sum(dOdC*item) for item in dCdp]

        cons = ({'type': 'ineq', 'fun': lambda x:  x},)

        res = minimize(_objfunc, x0=p0,
                       #jac=_objfunc_grad,
                       method='SLSQP',
                       constraints=cons)

        return res.x, res.success
        
    def hyperpar_mh_update(self, y, kpar_cur,
                           kpar_proposal,
                           kpar_prior,
                           lpcur=None,
                           store_new_par=False,
                           store_inv_cov=False,
                           **kwargs):
        """
        Performs a single Metropolis-Hastings update of the process hyper-parameters
        """
        if lpcur is None:
            lpcur = self.loglikelihood(y, kpar=kpar_cur)

        kpar_new, prop_ratio = kpar_proposal.rvs(kpar_cur, **kwargs)

        pnew = kpar_prior.pdf(kpar_new)
        if pnew > 0:

            lpnew, cnew = self.loglikelihood(y, kpar=kpar_new, return_cov=True)

            lpcur += kpar_prior.logpdf(kpar_cur)
            lpnew += kpar_prior.logpdf(kpar_new)

            A = np.exp(lpnew - lpcur)*prop_ratio
            if np.random.uniform() <= A:
                if store_new_par:
                    self.kernel.kpar = kpar_new
                    self.cov = cnew
                    if store_inv_cov:
                        self.inv_cov = np.linalg.inv(cnew)
                return kpar_new, lpnew, True

        return kpar_cur, lpcur, False


class GradientGaussianProcess:
    def __init__(self, kernel):
        assert(isinstance(kernel, GradientKernel))
        self.kernel = kernel
        self.is_mean_zero = True

    def fit(self, x=None, grad_x=None, y=None, dy=None, kpar=None):

        # Assert that one of x, grad_x is not None

        if kpar is None:
            kpar = self.kernel.kpar
        else:
            self.kernel.kpar = kpar

        self.x_inputs = x
        self.dx_inputs = grad_x

        if x is not None and grad_x is None:
            self.cov_x = self.kernel.cov(x)
            self.cov_x_inv = np.linalg.inv(self.cov_x)

        elif x is None:
            pass

        else:
            # we have input points for x and xdx
            print("Sup!")
            pass


##
# Gradient of a multivariate normal logpdf with respect
# to the covariance matrix
def _mvt_loglik_grad(y, m, cov):
    cov_inv = np.linalg.inv(cov)
    eta = y - m
    etaetaT = np.outer(eta, eta)
    expr = np.dot(cov_inv, np.dot(etaetaT, cov_inv))

    return -0.5*(cov_inv - expr)


########

class BaseGaussianProcess:

    def __init__(self, kernel, is_mean_zero=True):
        """
        Initalises me!
        """
        assert(isinstance(kernel, pydygp.kernels.BaseKernel))

        self.kernel = kernel
        self.is_mean_zero = is_mean_zero

    def fit(self, x, y=None, kpar=None,
            store_cov=False, store_inv_cov=True):
        """
        Carries out basic construction of the covariance functions for given kernel parameters
        """

        if kpar is None:
            kpar = self.kernel.kpar
        else:
            # kernel.kpar is set to kpar
            # for consistency
            self.kernel.kpar = kpar

        self.inputs = x
        self.outputs = y

        # creates the cov. matrix Cov(xi, xj)
        cov = self.kernel.cov(x)  

        if store_cov:
            self.cov = cov
        if store_inv_cov:
            self.inv_cov = np.linalg.inv(cov)

    def pred(self, xnew, return_covar=False):

        # bodge for predicingting new value at a single
        # point for a 1-dim kernel
        if isinstance(xnew, float):
            xnew = np.array([xnew]).reshape(1, 1)

        try:
            C22inv = self.inv_cov
        except:
            C22 = self.cov
            C22inv = np.linalg.inv(C22)

        C12 = self.kernel.cov(xnew, x2=self.inputs)
        C11 = self.kernel.cov(xnew)

        # conditional mean
        mcond = np.dot(C12, np.dot(C22inv, self.outputs))
        if return_covar:
            ccond = C11 - np.dot(C12, np.dot(C22inv, C12.T))
            return mcond, ccond

        else:
            return mcond

    def loglikelihood(self, y, x=None, kpar=None,
                      jitter=True, jitter_val=1e-5,
                      return_inf=True):

        if self.is_mean_zero:
            mean = np.zeros(y.size)

        if kpar is not None and x is None:
            cov = self.kernel.cov(self.inputs, kpar=kpar)

        elif kpar is not None and x is not None:
            cov = self.kernel.cov(x, kpar=kpar)

        elif kpar is None and x is not None:
            cov = self.kernel.cov(x)

        else:
            cov = self.cov

        if jitter:
            cov += np.diag(jitter_val*np.ones(cov.shape[0]))

        try:
            lp = multivariate_normal.logpdf(y, mean=mean, cov=cov)
        except:
            lp = -np.inf

        return lp

class BayesGaussianProcess(BaseGaussianProcess):
    def __init__(self, kern, priors=None):
        self.kpar_priors = priors
        super(BayesGaussianProcess, self).__init__(kern)

    def mcmc_fit(self, y, par0, nsim, nburn, nskip, proposals=None, is_symm_prop=False):
        self.proposals = proposals
        self.is_symm_prop = is_symm_prop

    def inv_scale_gibbs_update(self):
        # For those kernels which have an absolute scale
        # parameter, and for which the prior on the scale is
        # Gamma we can carry out a Gibbs update of the (inverse) scale
        pass


class MHBayesGaussianProcess(BayesGaussianProcess):

    def __init__(self, kern, priors, proposals, is_symm_prop=False):

        self.proposals = proposals
        self.is_symm_prop = is_symm_prop
        
        super(MHBayesGaussianProcess, self).__init__(kern, priors)

    # Attributes related to the current values of the
    # metropolis hastings sampler
    @property
    def kpar_cur(self):
        try:
            return self._kpar_cur
        except:
            self._kpar_cur = self.kernel.kpar
            return self._kpar_cur

    @property
    def kpar_cur_lp(self):
        try:
            return self._kpar_cur_lp
        except:
            lp = sum([prior.logpdf(p) for prior, p in zip(self.kpar_priors, self.kpar_cur)])
            self._kpar_cur_lp = lp
            return lp

    @property
    def cur_ll(self):
        try:
            return self._cur_ll
        except:
            ll = self.loglikelihood(self.y, self.x, kpar=self.kpar_cur)
            self._cur_ll = ll
            return ll

    def mh_update(self, cur_par):
        new_par = [prop.rvs(pcur) for prop, pcur in zip(self.proposals, cur_par)]

        lprior_cur = self.kpar_cur_lp
        lprior_new = sum([prior.logpdf(p) for prior, p in zip(self.kpar_priors, new_par)])

        ll_cur = self.cur_ll
        ll_new = self.loglikelihood(self.y, self.x, kpar=new_par)

        if self.is_symm_prop:
            lqratio = 0.
        else:
            raise NotImplementedError

        lA = ll_new - ll_cur  + lprior_new - lprior_cur + lqratio
        if np.log(np.random.uniform()) <= lA:

            self._kpar_cur = new_par
            self._kpar_cur_lp = lprior_new
            self._cur_ll = ll_new

class _GaussianProcess:
    """
    Gaussian Process class exposed to the user

    Behaves more like a factory to manage to the creation of
    application dependant Gaussian Process type objects
    """
    def __new__(cls, type, **kwargs):
        ktype, gptype = type.split("_")
        kern = _Kernel(ktype)

        if gptype == 'gpr':
            return BaseGaussianProcess(kern)

        elif gptype == 'mhbayes':
            raise NotImplementedError



def back_sub(L, x):
    return np.linalg.solve(L.T, np.linalg.solve(L, x))
