import numpy as np
import sklearn.gaussian_process.kernels as sklearn_kernels
from scipy.spatial.distance import pdist, cdist, squareform

def _check_length_scale(X, length_scale):
    length_scale = np.squeeze(length_scale).astype(float)
    if np.ndim(length_scale) > 1:
        raise ValueError("length_scale cannot be of dimension greater than 1")
    if np.ndim(length_scale) == 1 and X.shape[1] != length_scale.shape[0]:
        raise ValueError("Anisotropic kernel must have the same number of "
                         "dimensions as data (%d!=%d)"
                         % (length_scale.shape[0], X.shape[1]))
    return length_scale

class GradientKernel(sklearn_kernels.Kernel):
    """
    Base class for the gradient kernel.
    """
    def __mul__(self, b):
        if isinstance(b, GradientKernel):
            return GradientKernelProduct(self, b)
        else:
            raise ValueError("Multiplication must be between two GradientKernels")

    def __add__(self, b):
        raise NotImplementedError("Addition of GradientKernels not yet supported. Feel free to contribute!")

class GradientKernelProduct(sklearn_kernels.Product):

    def __call__(self, X, Y=None, eval_gradient=False, comp='x'):
        if comp == 'x':
            return super(GradientKernelProduct, self).__call__(X, Y, eval_gradient=eval_gradient)

        elif comp == 'xdx':
            if eval_gradient:
                K1, K1_gradient = self.k1(X, Y, eval_gradient=True)
                K1dx, K1dx_gradient = self.k1(X, Y, comp='xdx', eval_gradient=True)
                K2, K2_gradient = self.k2(X, Y, eval_gradient=True)
                K2dx, K2dx_gradient = self.k2(X, Y, comp='xdx', eval_gradient=True)

                # gradient wrt first par.
                grad1 = K1dx_gradient * K2[..., np.newaxis, np.newaxis] + \
                        K1_gradient[...,np.newaxis, :] * K2dx[..., np.newaxis]

                # gradient wrt second par.
                grad2 = K1dx[..., np.newaxis] * K2_gradient[..., np.newaxis, :] + \
                        K1[..., np.newaxis, np.newaxis] * K2dx_gradient

                Kdx = K1dx * K2[..., np.newaxis] + K1[..., np.newaxis] * K2dx
                Kdx_gradient = np.stack((grad1, grad2), axis=3)

                return Kdx, Kdx_gradient[...,0]
                
            else:
                K1 = self.k1(X, Y)
                K1dx = self.k1(X, Y, comp='xdx')
                K2 = self.k2(X, Y)
                K2dx = self.k2(X, Y, comp='xdx')
                return K1dx * K2[..., np.newaxis] + K1[..., np.newaxis] * K2dx

        elif comp == 'dxdx':
            if eval_gradient:
                K1, K1_gradient = self.k1(X, Y, eval_gradient=True)
                K1dx, K1dx_gradient = self.k1(X, Y, comp='xdx', eval_gradient=True)
                K1dxdx, K1dxdx_gradient = self.k1(X, Y, comp='dxdx', eval_gradient=True)
                K2, K2_gradient = self.k2(X, Y, eval_gradient=True)
                K2dx, K2dx_gradient = self.k2(X, Y, comp='xdx', eval_gradient=True)
                K2dxdx, K2dxdx_gradient = self.k2(X, Y, comp='dxdx', eval_gradient=True)

                grad1 = K1dxdx_gradient * K2[..., np.newaxis, np.newaxis, np.newaxis] + \
                        K1dx_gradient[:, :, np.newaxis, :, :] * \
                        K2dx[:, :, :, np.newaxis, np.newaxis] + \
                        K1dx_gradient[:, :, :, np.newaxis, :] * \
                        K2dx[:, :, np.newaxis, :, np.newaxis] + \
                        K1_gradient[..., np.newaxis, np.newaxis, :] * \
                        K2dxdx[..., np.newaxis]

                grad2 = K1dxdx[..., np.newaxis] * \
                        K2_gradient[..., np.newaxis, np.newaxis] + \
                        K1dx[:, :, np.newaxis, :, np.newaxis] * \
                        K2dx_gradient[:, :, :, np.newaxis, :] + \
                        K1dx[:, :, :, np.newaxis, np.newaxis] * \
                        K2dx_gradient[:, :, np.newaxis, :, :] + \
                        K1[..., np.newaxis, np.newaxis, np.newaxis] * \
                        K2dxdx_gradient

                Kdxdx_gradient = np.stack((grad1, grad2), axis=4)

                Kdxdx = K1dxdx * K2[..., np.newaxis, np.newaxis] + \
                        K1dx[..., np.newaxis, :] * K2dx[..., np.newaxis] + \
                        K1dx[..., np.newaxis] * K2dx[..., np.newaxis, :] + \
                        K1[..., np.newaxis, np.newaxis] * K2dxdx

                return Kdxdx, Kdxdx_gradient[..., 0]



            else:
                K1 = self.k1(X, Y)
                K1dx = self.k1(X, Y, comp='xdx')
                K1dxdx = self.k1(X, Y, comp='dxdx')
                K2 = self.k2(X, Y)
                K2dx = self.k2(X, Y, comp='xdx')
                K2dxdx = self.k2(X, Y, comp='dxdx')

                return K1dxdx * K2[..., np.newaxis, np.newaxis] + \
                       K1dx[..., np.newaxis, :] * K2dx[..., np.newaxis] + \
                       K1dx[..., np.newaxis] * K2dx[..., np.newaxis, :] + \
                       K1[..., np.newaxis, np.newaxis] * K2dxdx

class ConstantKernel(GradientKernel, sklearn_kernels.ConstantKernel):
    """Constant kernel.

    Can be used as part of a product-kernel where it scales the magnitude of
    the other factor (kernel) or as part of a sum-kernel, where it modifies
    the mean of the Gaussian process.

    For the gradient kernel higher order derivatives are zero.

    k(x_1, x_2) = constant_value for all x_1, x_2

    Parameters
    ----------
    constant_value : float, default: 1.0
        The constant value which defines the covariance:
        k(x_1, x_2) = constant_value

    constant_value_bounds : pair of floats >= 0, default: (1e-5, 1e-5)
        The lower and upper bound on constant_value

    """
    def __call__(self, X, Y=None, eval_gradient=False, comp='x'):
        if comp == 'x':
            return super(ConstantKernel, self).__call__(X, Y, eval_gradient)
        else:
            if Y is None:
                Y = X
            elif eval_gradient:
                raise ValueError

            if comp == 'xdx':
                if eval_gradient:
                    res = np.zeros((X.shape[0], X.shape[0], X.shape[1]))
                    return res, res[..., np.newaxis]
                else:
                    return np.zeros((X.shape[0], Y.shape[0], X.shape[1]))

            elif comp == 'dxdx':
                if eval_gradient:
                    res = np.zeros((X.shape[0], X.shape[0],
                                    X.shape[1], X.shape[1],))
                    return res, res[..., np.newaxis]
                else:
                    return np.zeros((X.shape[0], Y.shape[0],
                                     X.shape[1], X.shape[1]))

class RBF(GradientKernel, sklearn_kernels.RBF):

    def __call__(self, X, Y=None, eval_gradient=False, comp='x'):
        if comp == 'x':
            return super(RBF, self).__call__(X, Y, eval_gradient)

        else:


            X = np.atleast_2d(X)
            length_scale = _check_length_scale(X, self.length_scale)

            if Y is None:
                Y = X

            Diffs = (X[:, np.newaxis, :] - Y[np.newaxis, :, :])

            if eval_gradient:
                K, K_gradient = super(RBF, self).__call__(X, eval_gradient=True)
            else:
                K = np.exp(-.5 * np.sum(Diffs ** 2 / (length_scale ** 2), axis=2) )

            if comp == 'xdx':

                Kdx = (Diffs / (length_scale ** 2)  ) * K[..., np.newaxis]

                if eval_gradient:
                    Diffs /= (length_scale ** 2)
                    K, K_gradient = super(RBF, self).__call__(X, None, eval_gradient=True)

                    Kdx_gradient = Diffs[..., np.newaxis] * K_gradient[..., np.newaxis, :] 

                    # add a diagonal term contribution
                    for d in range(Kdx_gradient.shape[-1]):
                        Kdx_gradient[..., d, d] -= 2*Diffs[..., d]*K

                    return Kdx, Kdx_gradient

                else:
                    return Kdx

            elif comp == 'dxdx':

                Diffs /= (length_scale ** 2)
                Kdxdx = -Diffs[..., np.newaxis, :] * Diffs[..., np.newaxis]

                # add a diagonal term contribution
                if Diffs.shape[-1] == 1:
                    Kdxdx[:, :, 0, 0] += 1 / (length_scale ** 2)
                    
                else:
                    for d in range(Diffs.shape[-1]):
                        Kdxdx[:, :, d, d] += 1 / (length_scale[d] ** 2)

                if eval_gradient:
                    # Kdkdx = f * K
                    P = Diffs.shape[-1]
                    if P == 1:
                        length_scale = [length_scale]
                    
                    f = Kdxdx.copy()
                    f_gradient = np.zeros(Kdxdx.shape + (P, ))

                    for p in range(P):
                        f_gradient[..., p, p, p] -= 2 / (length_scale[p]**2)
                        for i in range(P):
                            f_gradient[...,p, i, p] += 2*Diffs[...,p] * \
                                                       Diffs[..., i]
                            f_gradient[...,i, p, p] += 2*Diffs[...,p] * \
                                                       Diffs[..., i]
                    expr1 = f_gradient * K[..., np.newaxis, np.newaxis, np.newaxis]

                    expr2 = f[..., np.newaxis] * \
                            K_gradient[:, :, np.newaxis, np.newaxis, :]

                    return Kdxdx * K[..., np.newaxis, np.newaxis], \
                           expr1 + expr2

                else:
                    return Kdxdx * K[..., np.newaxis, np.newaxis]
