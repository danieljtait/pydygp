import numpy as np
from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as sklearn_kernels
from scipy.special import wofz
from collections import namedtuple

_known_transform = {'log': lambda x: np.log(x),
                    'Id': lambda x: x}
_known_inv_transform = {'log': lambda x: np.exp(x),
                        'Id': lambda x: x}

class Hyperparameter(namedtuple('Hyperparameter',
                                ('name', 'value_type', 'bounds',
                                 'n_elements', 'fixed',
                                 'transform', 'inverse_transform'))):
    """A kernel hyperparamers's specification in form of a namedtuple.

    .. note
    An extension of the sklearn hyperparameter to allow the
    specifaction of the parameter transform
    """
    __slots__ = ()

    def __new__(cls, name, value_type, bounds, n_elements=1, fixed=None, transform='log'):
        bounds = np.atleast_2d(bounds)
        if n_elements > 1:  # vector-valued parameter
            if bounds.shape[0] == 1:
                bounds = np.repeat(bounds, n_elements, 0)
            elif bounds.shape[0] != n_elements:
                raise ValueError("Bounds on %s should have either 1 or "
                                 "%d dimensions. Given are %d"
                                 % (name, n_elements, bounds.shape[0]))
        if fixed is None:
            fixed = False
            #fixed = isinstance(bounds, six.string_types) and bounds == "fixed"

        try:
            transform, inv_transform = \
                       (_known_transform[transform], _known_inv_transform[transform])

        except KeyError:
            raise NotImplementedError("Unknown transform % s" % transform)
            
        return super(Hyperparameter, cls).__new__(
            cls, name, value_type, bounds, n_elements, fixed, transform, inv_transform)

    def __eq__(self, other):
        return (self, name == other.name and
                self.value_type == other.value_type and
                np.all(self.bounds == other.bounds) and
                self.n_elements == other.n_elements and
                self.fixed == other.fixed and
                self.transform == other.transform)

class Kernel(sklearn_kernels.Kernel):

    @property
    def theta(self):
        """ Returns the (flattened, 'transformed') non-fixed hyperparameters.
        
        Note that the transformation is specified during the initialisation of the kernel.
        """
        theta = []
        params = self.get_params()
        for hyperparameter in self.hyperparameters:
            if not hyperparameter.fixed:
                # transform the hyperparameter
                _theta = hyperparameter.transform(params[hyperparameter.name])
                theta.append(_theta)
        if len(theta) > 0:
            return np.hstack(theta)
        else:
            return np.array([])

    @theta.setter
    def theta(self, theta):
        params = self.get_params()
        i = 0
        for hyperparameter in self.hyperparameters:
            if hyperparameter.fixed:
                continue
            if hyperparameter.n_elements > 1:
                # vector-valued parameter
                par = hyperparameter.inverse_transform(theta[i:i + hyperparameter.n_elements])
                params[hyperparameter.name] = par
                i += hyperparameter.n_elements
            else:
                params[hyperparameter.name] = hyperparameter.inverse_transform(theta[i])
                i += 1

        if i != len(theta):
            raise ValueError("theta has not got the correct number of entries."
                             " Should be %d; given are %d"
                             % (i, len(theta)))

        self.set_params(**params)

    @property
    def bounds(self):
        """Returns the transformed bounds of theta.

        Returns
        -------
        bounds : array, shape (n_dims, 2)
        """
        bounds = []
        for hyperparameter in self.hyperparameters:
            if not hyperparameter.fixed:
                bounds.append(hyperparameter.transform(hyperparameter.bounds))
        if len(bounds) > 0:
            return (np.vstack(bounds))
        else:
            return np.array([])

def wofz_grad(z):
    return -2*z*wofz(z) + 1j*2/np.sqrt(np.pi)

def Upsilon(t, t2, gam, l_scale, eval_gradient=False):
    """
    Describe me without a reference?
    """
    # array axes ordering [ t , t2 , gamma, length scales]
    
    # pwd of times
    Dt = t[:, None] - t2[None, :]

    # u1 = 2exp(arg1)
    arg1 = .25*gam[:, None] **2 * l_scale[None, :] ** 2
    arg1 = arg1[None, None, ...] - \
           Dt[..., None, None] * gam[None, None, :, None]

    # u2 = exp(arg2) * wofz(1j * zrq)
    zrq = Dt[..., None, None] / l_scale - \
          .5 * gam[None, None, :, None] * l_scale  # easy b.-casting over last axis

    arg2 = - (Dt**2)[..., None, None] / l_scale**2

    # u3 = exp(arg3) * wofz(-1j * zrq0)
    zrq0 = -t2[None, :, None, None] / l_scale - \
           .5 * gam[None, None, :, None] * l_scale

    arg3 = -(t2**2)[None, :, None, None] / l_scale**2 -\
           t[:, None, None, None] * gam[None, None, :, None]

    if eval_gradient:

        # gradient of arg1 wrt gam
        darg1_g = .5*gam[:, None] * (l_scale ** 2)[None, :]
        darg1_g = darg1_g[None, None, ...] - \
                  Dt[..., None, None]

        u1 = 2 * np.exp(arg1) # for clarity.
        du1 = u1 * darg1_g

        u2 = np.exp(arg2) * wofz(1j * zrq)
        du2 = -.5*1j * wofz_grad(1j * zrq) * np.exp(arg2) * l_scale

        u3 = np.exp(arg3) * wofz(-1j*zrq0)
        du3 = np.exp(arg3) * \
              (-t[:, None, None, None] * wofz(-1j * zrq0) + \
               .5 * 1j * wofz_grad(-1j * zrq0) * l_scale)

        return u1 - u2 - u3, du1 - du2 - du3

    else:
        return 2*np.exp(arg1) - \
               np.exp(arg2)*wofz(1j*zrq) - \
               np.exp(arg3)*wofz(-1j*zrq0)

def Aicht2(t, t2, gq, gp, length_scale, eval_gradient=False):

    if eval_gradient:
        U1, U1grad = Upsilon(t2, t, gq, length_scale, True)
        # Add axis for gp
        U1 = U1[..., None, :]
        U1grad = U1grad[..., None, :]

        U2, U2grad = Upsilon(t2, np.zeros(t.size), gq, length_scale, True)
        # Add axis for gp
        U2 = U2[..., np.newaxis, :]
        U2grad = U2grad[..., np.newaxis, :]

        h = np.exp(-t[:, None]*gp[None, :])
        # - add axis for t2, gq, lr
        h = h[np.newaxis, :, np.newaxis, :, np.newaxis]
        dH = -h*t[None, :, None, None, None]

        H = U1 - h * U2
        gq_plus_gp = gq[:, None] + gp[None, :]
        # Add axis for t2, t, lr

        DH1grad = np.zeros((t2.size, t.size, gq.size, gp.size,
                            length_scale.size, gq.size),
                           dtype=np.complex128)
        DH2grad = np.zeros(DH1grad.shape,
                           dtype=np.complex128)

        for k in range(gq.size):
            DH1grad[..., k, :, :, k] += (U1grad[..., k, :, :] - 
                                         h[..., 0, :, :] * U2grad[..., k, :, :]) / \
                                         gq_plus_gp[None, None, k, :, None]

            DH1grad[..., k, :, :, k] -= H[..., k, :, :] / (gq_plus_gp[None, None, k, :, None])**2
            
            DH2grad[..., k, :, k] -= dH[..., k, :] * U2[..., 0, :] / \
                                     gq_plus_gp[None, None, :, k, None]

            DH2grad[..., k, :, k] -= H[..., k, :] / (gq_plus_gp[None, None, :, k, None])**2
                                     

        return H / gq_plus_gp[None, None, ..., None], \
               DH1grad, \
               DH2grad

    else:
        U1 = Upsilon(t2, t, gq, length_scale)[..., None, :]
        U2 = Upsilon(t2, np.zeros(t.size), gq, length_scale)[..., None, :]

        # Add axis for t2, gq, lr
        H = np.exp(-t[:, None] * gp[None, :])[None, :, None, :, None]
        H = U1 - H * U2
        
        gq_plus_gp = gq[:, None] + gp[None, :]
        # Add axis for t2, t, lr
        gq_plus_gp = gq_plus_gp[np.newaxis, np.newaxis, ..., np.newaxis]        

        return H / gq_plus_gp


class LFMorder2Kernel(Kernel):

    def __init__(self,
                 C=1.0, C_bounds=(-1e5, 1e5),
                 D=1.0, D_bounds=(-1e5, 1e5),
                 S=[1.], S_bounds=(-1e5, 1e5),
                 length_scale=1., length_scale_bounds=(1e-5, 1e5)):

        self.C = np.asarray(C)
        self.C_bounds = C_bounds
        self.S = np.asarray(S)
        self.S_bounds = S_bounds
        self.D = np.asarray(D)
        self.D_bounds = D_bounds
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds

        # some dimensionality checks
        if self.C.size == self.D.size:
            self.n_outputs = self.C.size
        else:
            raise ValueError("C and D must be the same size")
        n_inputs = self.S.size // self.C.size
        if n_inputs > 0:
            self.n_inputs = n_inputs
        else:
            raise ValueError("S must be of size n_inputs * n_outputs")

        if isinstance(length_scale, (int, float)):
            self.n_latentforces = 1
        else:
            self.n_latentforces = len(self.length_scale)

    def is_stationary(self):
        """ Returns whether the kernel is stationary. """
        return False

    @property
    def hyperparameter_C(self):
        return Hyperparameter(
            "C", "numeric", self.C_bounds, self.C.size, transform='Id')

    @property
    def hyperparameter_D(self):
        return Hyperparameter(
            "D", "numeric", self.D_bounds, self.D.size, transform='Id')

    @property
    def hyperparameter_S(self):
        return Hyperparameter(
            "S", "numeric", self.S_bounds,
            self.S.size, transform='Id')

    @property
    def hyperparameter_length_scale(self):
        if isinstance(self.length_scale, (float, int)):
            return Hyperparameter("length_scale", "numeric",
                                  self.length_scale_bounds, fixed=True)
        else:
            return Hyperparameter("length_scale", "numeric",
                                  self.length_scale_bounds,
                                  len(self.length_scale), fixed=True)
                                  
            

    def diag(self, X):
        """
        Should be possible to speed this up.
        """
        return np.diag(self(X)).copy()

    def __call__(self, X, Y=None, eval_gradient=False):
        """ Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array, shape (n_samples_X, 1)
            Left argument of the returned kernel k(X, Y)

        Y : array, shape (n_samples_Y, 1), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.

        Returns
        -------
        K : array, shape (n_samples_X * n_output_dim, 1)
            Kernel k(X, Y)

        K_gradient : array (opt.), shape (n_samples_X * n_output_dim,
        n_samples_X * n_output_dim, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True
        """            
        X = np.atleast_2d(X)
        if Y is None:
            Y = X.copy()

        # get the parameters
        parameters = self.get_params()
        C = parameters['C']
        D = parameters['D']
        S = parameters['S']
        length_scale = parameters['length_scale']
        length_scale = np.atleast_1d(length_scale)

        # get some dimensionality parameters...
        Q = self.n_outputs
        R = self.n_latentforces

        # slightly dirty hack to make
        # this kernel have an algebra that works with
        # the other kernels
        N1 = X.shape[0]# // Q
        N2 = Y.shape[0]# // Q

        t1 = X[:N1, 0]
        t2 = Y[:N2, 0]

        # and use them to reshape the vectorised S
        S = S.reshape((R, Q)).T


        w = 0.5*np.sqrt(4*D - C**2 + 1j*0)
        a = 0.5*C

        # gamma and its conjugate
        gam = a + 1j*w
        gamc = a - 1j*w

        # Sprod[..., r] = np.outer(S[:, r], S[:, r])
        Sprod = S[:, None, :] * S[None, :, :]
        wprod = np.outer(w, w)

        if eval_gradient:
            dadc = .5

            dwdc = -.5*C/np.sqrt(4*D - C**2 + 1j*0)
            dwdd = .5/w

            dgamdc = dadc + 1j*dwdc
            dgamdd = 1j*dwdd
            dgamcdc = dadc - 1j*dwdc
            dgamcdd = -1j*dwdd

            H11, D1H11, D2H11 = \
                 Aicht2(t1, t2, gamc, gam, length_scale, True)
            H12, D1H12, D2H12 = \
                 Aicht2(t1, t2, gam, gamc, length_scale, True)
            H13, D1H13, D2H13 = \
                 Aicht2(t1, t2, gamc, gamc, length_scale, True)
            H14, D1H14, D2H14 = \
                 Aicht2(t1, t2, gam, gam, length_scale, True)

            H21, D1H21, D2H21 = \
                 Aicht2(t2, t1, gam, gamc, length_scale, True)
            H22, D1H22, D2H22 = \
                 Aicht2(t2, t1, gamc, gam, length_scale, True)
            H23, D1H23, D2H23 = \
                 Aicht2(t2, t1, gamc, gamc, length_scale, True)
            H24, D1H24, D2H24 = \
                 Aicht2(t2, t1, gam, gam, length_scale, True)


            D_grad1 = np.zeros((N2, N1, Q, Q, R, Q), dtype=np.complex128)
            D_grad2 = np.zeros((N1, N2, Q, Q, R, Q), dtype=np.complex128)

            grad1 = D1H11 * dgamcdc + D2H11 * dgamdc + \
                    D1H12 * dgamdc + D2H12 * dgamcdc - \
                    D1H13 * dgamcdc - D2H13 * dgamcdc - \
                    D1H14 * dgamdc - D2H14 * dgamdc

            grad2 = D1H21 * dgamdc + D2H21 * dgamcdc + \
                    D1H22 * dgamcdc + D2H22 * dgamdc - \
                    D1H23 * dgamcdc - D2H23 * dgamcdc - \
                    D1H24 * dgamdc - D2H24 * dgamdc

            # gradient of H wrt D
            D_grad1 = D1H11 * dgamcdd + D2H11 * dgamdd + \
                      D1H12 * dgamdd + D2H12 * dgamcdd - \
                      D1H13 * dgamcdd - D2H13 * dgamcdd - \
                      D1H14 * dgamdd - D2H14 * dgamdd

            D_grad2 = D1H21 * dgamdd + D2H21 * dgamcdd + \
                      D1H22 * dgamcdd + D2H22 * dgamdd - \
                      D1H23 * dgamcdd - D2H23 * dgamcdd - \
                      D1H24 * dgamdd - D2H24 * dgamdd            

            

            H1 = H11 + H12 - H13 - H14
            # transpose the axis for H1
            H1 = H1.transpose((1, 0, 3, 2, 4))
            H2 = H21 + H22 - H23 - H24

            H = H1 + H2
            grad1 = grad1.transpose((1, 0, 3, 2, 4, 5))
            D_grad = D_grad1.transpose((1, 0, 3, 2, 4, 5)) + \
                     D_grad2            

            # [..., p, q, i, r] := d K_pq / d S_ir
            K_S_grad = np.zeros((N1, N2, Q, Q, Q, R), dtype=np.complex128)
            for p in range(Q):
                for q in range(Q):
                    K_S_grad[..., p, q, p, :] += (S[q, :] * np.sqrt(np.pi) * length_scale) * \
                                                 (H1[..., p, q, :] + H2[..., p, q, :])
                    K_S_grad[..., p, q, q, :] += (S[p, :] * np.sqrt(np.pi) * length_scale) * \
                                                 (H1[..., p, q, :] + H2[..., p, q, :])
            K_S_grad /= 8 * wprod[None, None, ..., None, None]
            K_S_grad = np.real(K_S_grad)            

            # [N1, N2, K, K, R, K]
            # Utility expression
            expr = Sprod * np.sqrt(np.pi) * length_scale / (8 * wprod[..., None])            
            expr_C_grad = np.zeros((Q, Q, Q), dtype=np.complex128)
            expr_D_grad = np.zeros((Q, Q, Q), dtype=np.complex128)
            for k in range(Q):
                expr_C_grad[k, :, k] -= dwdc[k] / (w[k]**2*w)
                expr_C_grad[:, k, k] -= dwdc[k] / (w[k]**2*w)
                expr_D_grad[k, :, k] -= dwdd[k] / (w[k]**2*w)
                expr_D_grad[:, k, k] -= dwdd[k] / (w[k]**2*w)

            # add an axis for the length scale
            expr_C_grad = expr_C_grad[:, :, None, :]
            expr_C_grad = expr_C_grad * (Sprod * np.sqrt(np.pi) * length_scale / 8)[..., None]

            expr_D_grad = expr_D_grad[:, :, None, :]
            expr_D_grad = expr_D_grad * (Sprod * np.sqrt(np.pi) * length_scale / 8)[..., None]


            K_C_grad = expr[None, None, ..., None] * (grad1 + grad2) + \
                       expr_C_grad[None, None, ...] * (H1 + H2)[..., None]

            K_D_grad = expr[None, None, ..., None] * (D_grad) + \
                       expr_D_grad[None, None, ...] * (H1 + H2)[..., None]

            K = expr[None, None, ...] * (H1 + H2)

            # sum over length scales
            K = K.sum(-1)
            K_C_grad = K_C_grad.sum(-2)
            K_D_grad = K_D_grad.sum(-2)

            # flatten everything out
            K = K.transpose((2, 0, 3, 1)).reshape((Q*N1, Q*N2))

            KdC = K_C_grad.transpose((2, 0, 3, 1, 4)).reshape((Q*N1, Q*N2, Q))
            KdD = K_D_grad.transpose((2, 0, 3, 1, 4)).reshape((Q*N1, Q*N2, Q))            
            KdS = K_S_grad.transpose((2, 0, 3, 1, 4, 5)).reshape((Q*N1, Q*N2, Q, R))

            # further flatten KdS to agree vec(S) = S.T.ravel()
            # [ S_11, ..., S_Q1, ..., ]
            KdS = np.stack([KdS[..., k, r] for r in range(R) for k in range(Q)],
                           axis=-1)

            K_grad = np.dstack((KdC, KdD, KdS))
            return np.real(K), np.real(K_grad)

        else:
            H1 = Aicht2(t1, t2, gamc, gam, length_scale) + \
                 Aicht2(t1, t2, gam, gamc, length_scale) - \
                 Aicht2(t1, t2, gamc, gamc, length_scale) - \
                 Aicht2(t1, t2, gam, gam, length_scale)
            H2 = Aicht2(t2, t1, gam, gamc, length_scale) + \
                 Aicht2(t2, t1, gamc, gam, length_scale) - \
                 Aicht2(t2, t1, gamc, gamc, length_scale) - \
                 Aicht2(t2, t1, gam, gam, length_scale)
            K = H1.transpose((1, 0, 3, 2, 4)) + H2

            expr = Sprod * np.sqrt(np.pi) * length_scale / (8 * wprod[..., None])
            K = K * expr[None, None, ...]
            K = K.sum(-1)

            K = np.real(K)
            # flatten K out
            K = K.transpose((2, 0, 3, 1)).reshape((Q*t1.size,
                                                   Q*t2.size))
            return K

class LFMorder2(GaussianProcessRegressor):
    
    
    def fit(self, X, y):

        self.kernel_ = clone(self.kernel)

        if self.normalize_y:
            self._y_train_mean = np.mean(y, axis=0)
            y = y - self._y_train_mean
        else:
            self._y_train_mean = np.zeros(1)

        if np.iterable(self.alpha) \
           and self.alpha.shape[0] != y.shape[0]:
            if self.alpha.shape[0] == 1:
                self.alpha = self.alpha[0]
            else:
                raise ValueError("alpha must be a scalar of an array"
                                 " with same number of entries as y (%d != %d)"
                                 % (self.alpha.shape[0], y.shape[0]))

        self.X_train_ = np.copy(X) if self.copy_X_train else X
        self.y_train_ = np.copy(y) if self.copy_X_train else y

        if self.optimizer is not None and self.kernel_.n_dims > 0:
            # Choose hyperparameters based on maximizing the log-marginal
            # likelihood (potentially starting from several initial values)
            def obj_func(theta, eval_gradient=True):
                try:
                    if eval_gradient:
                        lml, grad = self.log_marginal_likelihood(
                            theta, eval_gradient=True)
                        print(lml)
                        return -lml, -grad
                    else:
                        return -self.log_marginal_likelihood(theta)
                except KeyError:
                    return np.inf, np.zeros(theta.size)

            from scipy.optimize import minimize
            res = minimize(obj_func, x0=self.kernel_.theta, jac=True)

            self.kernel_ = self.kernel.clone_with_theta(res.x)
