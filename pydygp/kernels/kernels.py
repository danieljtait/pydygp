import numpy as np


class ParameterCollection:
    def __init__(self):
        pass

class KernelParameter:
    def __init__(self):
        pass

class Kernel:
    """
    .. note::

       An example of intersphinx is this: you **cannot** use :mod:`pickle` on this class.
    """
    def __init__(self, kfunc, kpar, **kwargs):
        self.kfunc = kfunc  # Callable giving cov{Y(x1), Y(x2)}
        self.kpar = kpar    # additional arguments to kfunc

        try:
            self.dim = kwargs['dim']
            self.cov_method = kwargs['cov_method']
            self.par_grad = kwargs['par_grad']
        except:
            pass


    def cov(self, x1, x2=None, kpar=None):
        if kpar is None:
            if isinstance(self.kpar, KernelParameter):
                kpar = self.kpar.get_value()
            elif isinstance(self.kpar, ParameterCollection):
                kpar = self.kpar.value()
            else:
                kpar = self.kpar

        if not isinstance(x1, np.ndarray):
            x1 = np.asarray(x1)

        if not isinstance(x2, (float, list, np.ndarray)):
            if x2 is None:
                x2 = x1.copy()
            else:
                raise ValueError("Unrecognised input to kernel covariance function")

        # Optionally supplied cov method takes precedence
        try:
            return self.cov_method(x1, x2, kpar)
        except:
            T, S = np.meshgrid(x2, x1)
            return self.kfunc(S.ravel(), T.ravel(), kpar,
                              **kwargs).reshape(T.shape)


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


    def __add__(self, other):
        if isinstance(other, Kernel):
            return AddKernel([self, other])
        elif isinstance(other, AddKernel):
            return AddKernel([self] + other.kernels)


    @classmethod
    def SquareExponKernel(cls, kpar=None, dim=1):
        """Create a square exponential kernel

        .. math::
           x^2 + y^2 = z^2
        """
        if not isinstance(kpar, np.ndarray):
            if kpar is None:
                kpar = np.ones(dim+1)
        if dim >= 1:
            def cov_method(xx1, xx2, par):
                xs = [np.meshgrid(x2, x1) for x1, x2 in zip(xx1.T, xx2.T)]
                exp_arg = sum(p*(x[0]-x[1])**2 for (x, p) in zip(xs, par[1:]))
                return par[0]*np.exp(-exp_arg)

            def par_grad(xx1, xx2, par, ind=-1):
                if ind < 0:
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
            
            return cls(None, kpar,
                       cov_method=cov_method,
                       dim=dim,
                       par_grad=par_grad)


    @classmethod
    def PeriodicKernel(cls, kpar=None):
        if not isinstance(kpar, np.ndarray):
            if kpar is None:
                kpar = np.ones(3)
            
        def cov_method(xx1, xx2, par):
            exp_arg = 2*np.sin(np.pi*np.abs(xx1.ravel()-xx2.ravel())/par[2])
            return par[0]**2*np.exp(-exp_arg/par[1]**2)

        return cls(None, kpar,
                   cov_method=cov_method,
                   dim=1)


class _SquareExponentialKernel(Kernel):
    """
    The square exponential kernel

    .. math::
       k(x, y) = \\theta_0 \exp\\left\\{ -\sum_{i=1}^D \\theta_i(x_i - y_i)^2 \\right\\}
    """
    def __init__(self, dim=1, kpar=None):
        self.dim = dim

        if not isinstance(kpar, np.ndarray):
            if kpar is None:
                kpar = np.ones(dim+1)
                
        if dim >= 1:
            def cov_method(xx1, xx2, par):
                xs = [np.meshgrid(x2, x1) for x1, x2 in zip(xx1.T, xx2.T)]
                exp_arg = sum(p*(x[0]-x[1])**2 for (x, p) in zip(xs, par[1:]))
                return par[0]*np.exp(-exp_arg)

            def par_grad(xx1, xx2, par, ind=-1):
                if ind < 0:
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
        
        super(SquareExponentialKernel, self).__init__(cov_method=cov_method,
                                                      dim=dim,
                                                      par_grad=par_grad)


class GradientKernel(Kernel):

    def __init__(self, kfunc, kpar, **kwargs):
        super(GradientKernel, self).__init__(kfunc, kpar, **kwargs)

    def cov(self,
            x1, x2=None, kpar=None,
            i=None, j=None, comp="x",
            **kwargs):

        if kpar is None:
            if isinstance(self.kpar, KernelParameter):
                kpar = self.kpar.get_value()
            elif isinstance(self.kpar, ParameterCollection):
                kpar = self.kpar.value()
            else:
                kpar = self.kpar

        if not isinstance(x1, np.ndarray):
            x1 = np.asarray(x1)

        if not isinstance(x2, (float, list, np.ndarray)):
            if x2 is None:
                x2 = x1.copy()
            else:
                raise ValueError("x2 input should be array-like")

        # Optionally supplied cov method takes precedence
        try:

            if comp == 'dxdx' and i is None and j is None:

                # ToDo - ...
                #
                cov = np.row_stack((
                    np.column_stack((self.cov_method(x1, x2, kpar,
                                                     i=_i, j=_j,
                                                     comp=comp)
                                     for _j in range(self.dim)))
                    for _i in range(self.dim)))
                return cov

            elif comp == 'xdx' and i is None:
                cov = np.column_stack((
                    self.cov_method(x1, x2, kpar,
                                    i=_i, comp=comp)
                    for _i in range(self.dim)))
                return cov

            elif comp == 'dxx' and i is None:
                cov = np.row_stack((
                    self.cov_method(x1, x2, kpar,
                                    i=_i, comp=comp)
                    for _i in range(self.dim)))
                return cov
            
            else:
                return self.cov_method(x1, x2, kpar,
                                       i=i, j=j, comp=comp)

        except:
            pass

    
    @classmethod
    def SquareExponKernel(cls, kpar=None, dim=1):
        if not isinstance(kpar, (np.ndarray, list)):
            if kpar is None:
                kpar = np.ones(dim+1)

        def _k(xx1, xx2, par):
            xs = [np.meshgrid(x2, x1) for x1, x2 in zip(xx1.T, xx2.T)]
            exp_arg = sum(p*(x[0]-x[1])**2 for (x, p) in zip(xs, par[1:]))
            return par[0]*np.exp(-exp_arg)

        def _kxdx(i, xx1, xx2, par):

            xs = [np.meshgrid(x2, x1) for x1, x2 in zip(xx1.T, xx2.T)]
            exp_arg = sum(p*(x[0]-x[1])**2 for (x, p) in zip(xs, par[1:]))

            k = par[0]*np.exp(-exp_arg)
            expr = 2*par[i+1]*(xs[i][1]-xs[i][0])

            return expr*k

        def _kdxdx(i, j, xx1, xx2, par):
            xs = [np.meshgrid(x2, x1) for x1, x2 in zip(xx1.T, xx2.T)]
            exp_arg = sum(p*(x[0]-x[1])**2 for (x, p) in zip(xs, par[1:]))
            _k = par[0]*np.exp(-exp_arg)

            if i != j:
                expr = -4*par[0]*par[i+1]*par[j+1]
                expr *= (xs[i][1]-xs[i][0])*(xs[j][1]-xs[j][0])
                return expr*_k
            else:
                expr1 = 2*par[i+1]*_k
                expr2 = 2*par[i+1]*(xs[i][1]-xs[i][0])
                expr2 *= -2*par[i+1]*(xs[i][1]-xs[i][0])*_k
                return expr1 + expr2

        def cov_method(xx1, xx2, par, i=None, j=None, comp='x'):
            if comp == 'x':
                return _k(xx1, xx2, par)
            elif comp == 'xdx':
                return _kxdx(i, xx1, xx2, par)
            elif comp == 'dxx':
                return -_kxdx(i, xx1, xx2, par)
            elif comp == 'dxdx':
                return _kdxdx(i, j, xx1, xx2, par)

        return cls(None, kpar,
                   cov_method=cov_method,
                   dim=dim)


class AddKernel:
    def __init__(self, kernels):
        self.kernels = kernels


    def cov(self, x1, x2=None):

        if not isinstance(x1, np.ndarray):
            x1 = np.asarray(x1)

        if not isinstance(x2, (float, list, np.ndarray)):
            if x2 is None:
                x2 = x1.copy()
            else:
                raise ValueError("Unrecognised input to kernel covariance function")

        return sum(kernel.cov(x1, x2) for kernel in self.kernels)

    def __add__(self, other):
        if isinstance(other, Kernel):
            return AddKernel(self.kernels + [other])
        elif isinstance(other, AddKernel):
            return AddKernel(self.kernels + other.kernels)



###### Slight rewriting of everything
class BaseKernel:
    """
    base class for a parameterised kernel function
    """
    def __init__(self, cov_method, kpar, par_grad=None):
        self.cov_method = cov_method
        self.kpar = kpar
        self.par_grad = par_grad

    def cov(self, x1, kpar=None, x2=None):
        """
        Returns the covariance matrix
        """
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


class SquareExponentialKernel(BaseKernel):

    def __init__(self, dim=1, kpar=None):
        """
        The Square Exponential Kernel with parameterisation

        .. math::
           k(x, y) = \\theta_0 \exp\\left\\{-\\sum_{k=1}^D \\theta_k (x_k - y_k)^2 \\right\\}.
        """

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


class _Kernel:
    def __new__(cls, ktype, *args, **kwargs):
        if ktype == 'sqexp':
            return SquareExponentialKernel(*args, **kwargs)
