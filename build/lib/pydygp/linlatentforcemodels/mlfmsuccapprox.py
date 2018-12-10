import numpy as np
import scipy.sparse as sparse
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.linalg import block_diag, cho_solve, cholesky
from scipy.optimize import minimize
from .mlfm import BaseMLFM, Dimensions
from collections import namedtuple
from . import util

MLFMSuccApproxFitResults = namedtuple('MLFMSuccApproxFitResults',
                                      'g, beta, logpsi, logphi, loggamma, logtau, logalpha, \
                                      optimres')


class BaseMLFMSuccApprox(BaseMLFM):

    is_tt_aug = False

    def __init__(self, *args, order=1, **kwargs):
        super(BaseMLFMSuccApprox, self).__init__(*args, **kwargs)

        if isinstance(order, int) and order > 0:
            self.order = order
        else:
            raise ValueError("The approximation order must be a"
                             " positive integer")

    def _setup_latentstates(self, kernels=None, **kwargs):
        """
        Create the initial GP approximation
        """
        if kernels is None:
            ls_kernels = [ConstantKernel()*RBF() for k in range(self.dim.K)]
            self.latentstates = [GaussianProcessRegressor(kern)
                                 for kern in ls_kernels]
        else:
            self.latentstates = [GaussianProcessRegressor(kern) for kern in ls_kernels]

    def _setup_times(self, tt, tt_aug=None, ifix=0, **kwargs):
        if tt_aug is not None:
            self.ttc = tt_aug.copy()
            self._tt_data = tt.copy()
            self.data_inds = kwargs['data_inds']
            self._Ndata = len(self.data_inds)
            self.is_tt_aug = True
        else:
            self.ttc = tt
        self.ifix = ifix
        self.dim = Dimensions(self.ttc.size, self.dim.K, self.dim.R, self.dim.D)
        self._weight_matrix = weight_matrix(self.ttc, ifix)

    def _setup(self, times, **kwargs):
        """Prepares the model for fitting
        """
        if not hasattr(self, 'ttc'):
            self._setup_times(times, **kwargs)

        if not hasattr(self, 'latentstates'):
            # no latent states supplied so use default setting
            self._setup_latentstates(**kwargs)

    def _K(self, g, beta, ifix=None):
        """ Returns the discretised integral operator
        """
        if ifix is None:
            ifix = self.ifix
    
        g = g.reshape(self.dim.R, self.dim.N).T        # set g as g_{nr}, n=1..N, r=1..R
        g = np.column_stack((np.ones(self.dim.N), g))  # append a col. of 1s to g

        # matrices A[r] = sum(b_rd * Ld
        struct_mats = np.array([sum(brd*Ld for brd, Ld in zip(br, self.basis_mats))
                                for br in beta])
        # Add an axis for n=0,...,N-1
        At = struct_mats[None, :, :, :] * g[..., None, None]
        At = At.sum(1)

        W = weight_matrix(self.ttc, ifix) #self._weight_matrix
        K = sum(sparse.kron(sparse.kron(W[:, i][:, None],
                                        sparse.eye(1, self.dim.N, i)),
                            At[i, ...])
                for i in range(self.dim.N))
        I = sparse.eye(self.dim.K)

        # because x[ifix] is fixed by the integral transform
        # we need to add a column of identity matrices to K
        # - could do this by assignment
        eifix = sparse.eye(1, self.dim.N, ifix)
        K += sum(sparse.kron(sparse.kron(sparse.eye(1, self.dim.N, i).transpose(),
                                         eifix),
                             I)
                 for i in range(self.dim.N))
        return K

    def _K2(self, g, beta, ifix=None):
        """ K acting on vec(X) rather than ravel(X)
        """
        K = self._K(g, beta, ifix)
        T = util.T_xrav_tovecx(self.dim.N, self.dim.K)
        Tt = T.transpose()
        return T.dot(K.dot(Tt))

    def get_weight_matrix(self, tt, i):
        return weight_matrix(self.ttc, i)

    def _K3(self, g, beta, ifix, printit=False):
        W = weight_matrix(self.ttc, ifix)

        g = g.reshape(self.dim.R, self.dim.N).T
        g = np.column_stack((np.ones(self.dim.N), g))

        # matrices A[r] = sum(b_rd * Ld
        struct_mats = np.array([sum(brd*Ld for brd, Ld in zip(br, self.basis_mats))
                                for br in beta])
        # Add an axis for n=0,...,N-1
        At = struct_mats[None, :, :, :] * g[..., None, None]
        At = At.sum(1)

        eifix = np.eye(1, self.dim.N, ifix)
        K = sum(np.kron(At[i,...],
                        np.kron(W[:, i][:, None], np.eye(1, self.dim.N, i)))
                        for i in range(self.dim.N))

        I = np.eye(self.dim.K)
        eifix = np.eye(1, self.dim.N, ifix)
        K += sum(np.kron(I,
                         np.kron(np.eye(1, self.dim.N, i).T, eifix))
                 for i in range(self.dim.N))
        return K

    def sparse_vecK_aff_rep(self, beta):
        """Returns the vectorisation of the discretised integral
        operator
        """
        V0 = self._K(np.zeros(self.dim.N*self.dim.R), beta)  # lazy way of finding K[0]
        W = self._weight_matrix

        # A[r] = sum_{brd} * Ld
        struct_mats = np.array([sum(brd*Lrd for brd, Lrd in zip(br, self.basis_mats))
                                for br in beta])

        # columns of dK.dot(g) + vec(V0) = vec(K[g, beta])
        NK = self.dim.N*self.dim.K
        K = self.dim.K

        cols = []
        for r in range(self.dim.R):
            for n in range(self.dim.N):
                v1 = sparse.csr_matrix((n*K*NK, 1))          # Padding of zeros either
                v3 = sparse.csr_matrix(((NK-(n+1)*K)*NK,1))  # side of the vector
                v2 = np.kron(W[:, n][:, None], struct_mats[r+1]).T.ravel()[:, None]
                cols.append(sparse.vstack((v1, v2, v3)))

        V = sparse.hstack(cols)
        v0 = sparse.coo_matrix(V0.todense().T.ravel()).transpose()  # ugly step

        # convert V, v0 to sparse csr (quick for .dot( some_dense_arr )
        return V.tocsr(), v0.tocsr()

    @property
    def Ndata(self):
        if self.is_tt_aug:
            return len(self.data_inds)
        else:
            return self.dim.N


class MLFMSAFitMixin:
    """
    Utility mixin for initialising fit functions
    """
    def _fit_kwarg_parser(self, **kwargs):
        """
        Parses kwargs to determine which var. to be kept fixed during optimisation.
        """
        # possibly makes these properties of a class
        var_names = ['g', 'beta', 'logpsi', 'logphi', 'loggamma', 'logtau', 'logalpha']

        # controls the default fixing of parameters
        default = {'g': False, 'beta': True, 'logpsi': True,
                   'logphi': True, 'loggamma': False, 'logtau': True, 'logalpha': True}
        
        # check **kwargs to see if any variables have been kept fixed
        is_fixed_vars = [kwargs.pop("".join((vn, "_is_fixed")),
                                    default[vn])
                         for vn in var_names]

        return is_fixed_vars

    def _fit_init(self, is_fixed_vars, **kwargs):
        """
        Handles initialisation of optimisation
        """
        init_strategies = {
            'g': lambda : np.zeros(self.dim.N*self.dim.R),
            'logpsi': lambda : np.concatenate([gp.kernel.theta
                                               for gp in self.latentforces]),
            'beta': lambda : np.row_stack((np.zeros(self.dim.D),
                                           np.eye(self.dim.R, self.dim.D))).ravel(),
            'logphi': lambda : np.concatenate([gp.kernel.theta
                                               for gp in self.latentstates]),
            'loggamma': lambda : np.log(1e-4*np.ones(self.dim.K)),
            'logtau': lambda : np.log(1e4*np.ones(self.dim.K)),
            'logalpha': lambda : np.zeros(self.dim.K),
            }

        var_names = ['g', 'beta', 'logpsi', 'logphi', 'loggamma', 'logtau', 'logalpha']
        full_init = [kwargs.pop("".join((vn, '0')),
                                init_strategies[vn]())
                     for vn in var_names]
        full_init_shape = [item.size for item in full_init]

        free_vars, fixed_vars = ([], [])
        for item, boolean in zip(full_init, is_fixed_vars):
            if boolean:
                fixed_vars.append(item)
            else:
                free_vars.append(item)
        free_vars_shape = [item.size for item in free_vars]

        return np.concatenate(free_vars), free_vars_shape, fixed_vars

    def _fit_find_priors(self, **kwargs):
        priors = {}
        for vn in ['logpsi', 'beta', 'logtau', 'loggamma', 'logalpha']:
            key = '_'.join((vn, 'prior'))
            try:
                priors[vn] = kwargs[key].logpdf
            except KeyError:
                pass
        return priors

    def _var_mixer(self, free_vars, free_vars_shape, fixed_vars, is_fixed_vars):
        """ Utility function to mix the free vars and the fixed vars
        """
        free_vars = util._unpack_vector(free_vars, free_vars_shape)
        if is_fixed_vars is None:
            return free_vars
        else:
            full_vars = []
            ifree, ifixed = (0, 0)
            for b in is_fixed_vars:
                if b:
                    full_vars.append(fixed_vars[ifixed])
                    ifixed += 1
                else:
                    full_vars.append(free_vars[ifree])
                    ifree += 1
            return full_vars

class PredictMixin:
    """
    Extends the capabilties to behave like the mutliout GP
    """
    def predict_lf(self, tt, return_std=False):
        Cff_ = block_diag(*(gp.kernel_(tt[:, None], self.X_train_)
                            for gp in self.latentforces))
        lf_mean = Cff_.dot(self.g_alpha_)
        lf_mean = lf_mean.reshape(self.dim.R, tt.size).T

        if return_std:
            A = Cff_.dot(cho_solve((self.g_L_, True), np.eye(self.dim.N*self.dim.R)))
            K = A.dot(self.g_laplace_cov.dot(A.T))
            Cp = []
            for gp in self.latentforces:
                C22 = gp.kernel_(tt[:, None])
                C21 = gp.kernel_(tt[:, None], self.X_train_)
                Cp.append(C22 - C21.dot(cho_solve((gp.L_, True), C21.T)))
            K += block_diag(*Cp)
            return lf_mean, np.sqrt(np.diag(K))

        return lf_mean

class MLFMSuccApprox(PredictMixin, MLFMSAFitMixin, BaseMLFMSuccApprox):

    def __init__(self, *args, **kwargs):
        super(MLFMSuccApprox, self).__init__(*args, **kwargs)

    def fit(self, times, Y, **kwargs):
        """ Carries out MAP optimisation of the model parameters.
        """
        def objfunc(arg, free_vars_shape, fixed_vars):
            g, beta, logpsi, logphi, loggamma, logtau, logalpha = \
               self._var_mixer(arg, free_vars_shape, fixed_vars, is_fixed_vars)

            # reshape beta
            beta = beta.reshape(self.dim.R+1, self.dim.D)

            try:
                ll, ll_g_grad, ll_lgam_grad, ll_lalf_grad = \
                    self.log_likelihood(Y, g, beta,
                                        logphi, loggamma, logtau, logalpha,
                                        eval_gradient=True)
                lp, lp_g_grad = \
                    self.prior_logpdf(g, logpsi,
                                      eval_gradient=True)

                grad = [ll_g_grad + lp_g_grad,
                        None,
                        None,
                        None,
                        ll_lgam_grad,
                        None,
                        ll_lalf_grad]

                # add contributions from various priors
                for vn, indx in zip(['loggamma'],
                                    [(4, loggamma),]):
                    try:
                        prior_logpdf = priors[vn]
                        ind, x = indx  # unpack the value of var. name
                        vn_lp, vn_lp_grad = prior_logpdf(x, True)
                        lp += vn_lp
                        grad[ind] += vn_lp_grad
                    except KeyError:
                        pass

                grad = np.concatenate([item
                                       for item, b in zip(grad, is_fixed_vars)
                                       if not b])
                return -(ll+lp), -grad
                
            except:
                return np.inf, np.zeros(arg.shape)

        
        # make sure the model is ready for fitting by calling _setup
        self._setup(times, **kwargs)

        # parse  kwargs to see if any args to be kept fixed
        is_fixed_vars = self._fit_kwarg_parser(**kwargs)

        # Check for any priors
        priors = self._fit_find_priors(**kwargs)

        init, free_vars_shape, fixed_vars = \
              self._fit_init(is_fixed_vars, **kwargs)

        res = minimize(objfunc, init, jac=True,
                       args=(free_vars_shape, fixed_vars))

        # unpack the results
        g_, beta_, logpsi_, logphi_, loggamma_, logtau_, logalpha_ = \
            self._var_mixer(res.x, free_vars_shape, fixed_vars, is_fixed_vars)
        logpsi_ = util._unpack_vector(logpsi_,
                                      util._get_gp_theta_shape(self.latentforces))

        for r, gp in enumerate(self.latentforces):
            gp.kernel_ = gp.kernel.clone_with_theta(logpsi_[r])
            Cr = gp.kernel_(self.ttc[:, None])
            Cr[np.diag_indices_from(Cr)] += 1e-5
            gp.L_ = cholesky(Cr, True)

        
        # store some variables for later use
        self.X_train_ = self.ttc[:, None]
        if not is_fixed_vars[0]:
            # Optimised g - so get Laplace approx to from optim
            self.g_laplace_cov = res.hess_inv[:self.dim.N*self.dim.R,
                                              :self.dim.N*self.dim.R]
            Cff = block_diag(*(gp.kernel_(self.X_train_)
                               for gp in self.latentforces))
            Cff[np.diag_indices_from(Cff)] += 1e-4
            self.g_L_ = cholesky(Cff, lower=True)
            self.g_alpha_ = cho_solve((self.g_L_, True), g_)

        return MLFMSuccApproxFitResults(g_.reshape(self.dim.R, self.dim.N),
                                        beta_, logpsi_, logphi_, loggamma_, logtau_, logalpha_,
                                        res)

    def marginal_covar_matrix(self, g, beta, gamma, logphi, alpha, eval_gradient=False):
        """
        Covariance matrix K^n C0 K^nT
        """
        # reshape g
        g = g.reshape(self.dim.R, self.dim.N).T        # reshape 
        g = np.column_stack((np.ones(self.dim.N), g))  # and append a col. of 1s

        # get the struct matrices
        struct_mats = np.array([sum(brd*Ld for brd, Ld in zip(br, self.basis_mats))
                                for br in beta])
        # Add an axis for n=0,...,N-1
        At = struct_mats[None, :, :, :] * g[..., None, None]
        At = At.sum(1)

        # Get the additive noise error term
        #Gamma = np.diag(np.concatenate([gk*np.ones(self.dim.N) for gk in gamma]))
        Gamma = np.kron(np.eye(self.dim.N), np.diag(gamma))

        # Get the initial state covariance matrix
        C = np.kron(np.ones((self.dim.N, self.dim.N)),
                    np.diag(alpha))
        
        ####
        # Construct the transformation matrix K
        W = self._weight_matrix
        K = sum(sparse.kron(sparse.kron(W[:, i][:, None],
                                        sparse.eye(1, self.dim.N, i)),
                            At[i, ...])
                for i in range(self.dim.N))
        I = sparse.eye(self.dim.K)

        eifix = sparse.eye(1, self.dim.N, self.ifix)
        K += sum(sparse.kron(sparse.kron(sparse.eye(1, self.dim.N, i).transpose(),
                                         eifix),
                             I)
                 for i in range(self.dim.N))
        #_K = np.array(K.todense())
        if eval_gradient:
            dCdg = [np.zeros((self.dim.N*self.dim.K, self.dim.N*self.dim.K))]*self.dim.N*self.dim.R            

            #_dCdg = np.zeros((self.dim.N*self.dim.R,
            #                  self.dim.N*self.dim.K,
            #                  self.dim.N*self.dim.K))

            # Gradient of the operator K wrt g_rn
            dK = [sparse.kron(sparse.kron(W[:, n][:, None],
                                          sparse.eye(1, self.dim.N, n)),
                              struct_mats[r+1])
                  for r in range(self.dim.R)
                  for n in range(self.dim.N)]
            #_dK = np.array([np.kron(W[:, n][:, None], struct_mats[r+1])
            #                for r in range(self.dim.R)
            #                for n in range(self.dim.N)])

            dCdgamma = np.zeros((self.dim.N*self.dim.K,
                                 self.dim.N*self.dim.K,
                                 self.dim.K))
            dCdalpha = [np.kron(np.ones((self.dim.N, self.dim.N)),
                                np.diag(np.eye(N=1, M=self.dim.K, k=k).ravel()))
                        for k in range(self.dim.K)]

            for m in range(self.order):

                dCdg = np.array([K.dot(K.dot(dC_grn).T) for dC_grn in dCdg])          
                for i, dK_grn in enumerate(dK):
                    kcdk = K.dot(dK_grn.dot(C).T)
                    dCdg[i] += kcdk + kcdk.T

                # update gradient of C wrt to g_{rn}
                #CKt = K.dot(C).T
                #print(K.shape, _dCdg.shape)
                #expr = _K.dot(_dCdg)
                #print(expr.shape)
                #assert(False)
                #expr = dK.dot(CKt.reshape(self.dim.N, self.dim.K, self.dim.N*self.dim.K)).sum(2)
                #expr += expr.transpose(0, 2, 1)
                #expr2 = K.dot(K.dot(dCdg).T)
                #dCdg = expr + expr2
                #dCdg = [K.dot(dK_grn.dot(C).T) + \
                #        dK_grn.dot(K.dot(C).T) + \
                #        K.dot(K.dot(dC_grn).T)
                #        for dK_grn, dC_grn in zip(dK, dCdg)]

                # update gradient of C wrt to Gamma
                dCdgamma = np.dstack((K.dot(K.dot(dCdgamma[..., i]).T)
                                      for i in range(dCdgamma.shape[-1])))

                diag_ek = np.zeros((self.dim.K, self.dim.K))
                for k in range(self.dim.K):
                    diag_ek[k, k] = 1.
                    dCdgamma[..., k] += np.kron(np.eye(self.dim.N), diag_ek)
                    diag_ek[k, k] = 0.

                # update gradient of C wrt to alpha
                dCdalpha = [K.dot(K.dot(dCdak).T) for dCdak in dCdalpha]
                
                C = K.dot(K.dot(C).T) + Gamma

            #print("...done.")
            dCdg = np.dstack(dCdg)
            return (C,
                    dCdg,
                    dCdgamma,
                    np.dstack(dCdalpha))

        else:
            for m in range(self.order):
                C = K.dot(K.dot(C).T) + Gamma            
            return C

    def marginal_covar_matrix2(self, g, beta, gamma, logphi, alpha, eval_gradient=False):
        """
        Rewrite to prevent some of the looping
        """
        g = g.reshape(self.dim.R, self.dim.N).T        # reshape 
        g = np.column_stack((np.ones(self.dim.N), g))  # and append a col. of 1s

        # get the struct matrices
        struct_mats = np.array([sum(brd*Ld for brd, Ld in zip(br, self.basis_mats))
                                for br in beta])
        # Add an axis for n=0,...,N-1
        At = struct_mats[None, :, :, :] * g[..., None, None]
        At = At.sum(1)

        # Get the additive noise error term
        Gamma = np.kron(np.eye(self.dim.N), np.diag(gamma))

        # Get the initial state covariance matrix
        C = np.kron(np.ones((self.dim.N, self.dim.N)),
                    np.diag(alpha))
        
        ####
        # Construct the transformation matrix K
        W = self._weight_matrix
        K = sum(sparse.kron(sparse.kron(W[:, i][:, None],
                                        sparse.eye(1, self.dim.N, i)),
                            At[i, ...])
                for i in range(self.dim.N))
        I = sparse.eye(self.dim.K)

        eifix = sparse.eye(1, self.dim.N, self.ifix)
        K += sum(sparse.kron(sparse.kron(sparse.eye(1, self.dim.N, i).transpose(),
                                         eifix),
                             I)
                 for i in range(self.dim.N))
        
        if eval_gradient:
            # dK[r, n, ...] = wn (x) Ar
            dK = [sparse.kron(sparse.kron(W[:, n][:, None],
                                          sparse.eye(1, self.dim.N, n)),
                              struct_mats[r+1])
                  for r in range(self.dim.R)
                  for n in range(self.dim.N)]

            dCdg = np.zeros((self.dim.N*self.dim.R,
                             self.dim.N*self.dim.K,
                             self.dim.N*self.dim.K))

            dCdalpha = [np.kron(np.ones((self.dim.N, self.dim.N)),
                                np.diag(np.eye(N=1, M=self.dim.K, k=k).ravel()))
                        for k in range(self.dim.K)]

            K = K.todense()
            Kt = np.array(K.transpose())
            
            for m in range(self.order):
                CKt = C.dot(Kt)

                # Update gradient of C wrt to g_{rn}
                for i, (dK_rn, dC_grn) in enumerate(zip(dK, dCdg)):
                    a = dK_rn.dot(CKt)
                    dCdg[i] = a + a.T + K.dot(dC_grn.dot(Kt))

                # Update gradient of C wrt to alpha
                dCdalpha = [K.dot(dCdak.dot(Kt))
                            for dCdak in dCdalpha]

                # Finally update C
                C = K.dot(CKt) + Gamma

            return C, dCdg, np.stack(dCdalpha)


    def data_cov(self, tau, logphi):
        Kdat = np.diag((np.ones((self.Ndata, self.dim.K))*(1/tau)).ravel())
        return Kdat
        logphi = util._unpack_vector(logphi,
                                     util._get_gp_theta_shape(self.latentstates))
        if self.is_tt_aug:
            tt = self._tt_data
        else:
            tt = self.ttc
        datacov = [gp.kernel.clone_with_theta(theta)(tt[:, None])
                   for gp, theta in zip(self.latentstates, logphi)]
        for item in datacov:
            item /= tau[0]
        datacov = block_diag(*datacov)
        #for k, C in enumerate(datainvcov):
        #    C[np.diag_indices_from(C)] += self.latentstates[k].alpha
        #datainvcov = [np.linalg.cholesky(C) for C in datainvcov]
        #datainvcov = block_diag(*[cho_solve((L, True), np.eye(L.shape[0]))
        #for L in datainvcov])
        # this is the inverse of Cov{vec(X)}, we need
        # Cov{rav(X)} = T Cov{vec(X)} Tt
        #   ---> InvCov{rav(X)} = Tt InvCov{vec(X)} T
        T = util.T_xrav_tovecx(self.Ndata, self.dim.K)
        #datainvcov = Tt.dot(Tt.dot(datainvcov).T)
        #Kdat += T.dot(T.dot(datacov).T)
        Kdat = datacov

        return Kdat
        
    def log_likelihood(self, y, g, beta,
                       logphi, loggamma, logtau, logalpha,
                       eval_gradient=False, **kwargs):
        if eval_gradient:
            K, K_g_grad, K_gamma_grad, K_alpha_grad = \
               self.marginal_covar_matrix(g, beta, np.exp(loggamma), logphi, np.exp(logalpha), True)
        else:
            K = self.marginal_covar_matrix(g, beta, np.exp(loggamma), logphi, np.exp(logalpha))

        if self.is_tt_aug:
            data_inds = np.asarray(self.data_inds)
            data_inds = (data_inds*self.dim.K)[:, None] + np.arange(self.dim.K)[None, :]
            data_inds_ix_ = np.ix_(data_inds.ravel(), data_inds.ravel())

            K = K[data_inds_ix_]
            if eval_gradient:
                K_g_grad = np.dstack([K_g_grad[..., i][data_inds_ix_]
                                      for i in range(K_g_grad.shape[-1])])
                K_gamma_grad = np.dstack([K_gamma_grad[..., i][data_inds_ix_]
                                          for i in range(K_gamma_grad.shape[-1])])
                K_alpha_grad = np.dstack([K_alpha_grad[..., i][data_inds_ix_]
                                          for i in range(K_alpha_grad.shape[-1])])

        # data covariance
        tau = np.exp(logtau)
        Kdat = self.data_cov(tau, logphi)
        K += Kdat

        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return (-np.inf, np.zeros_like(g.size)) \
                   if eval_gradient else -np.inf

        if len(y.shape) == 1:
            y_train = y[:, None]
        else:
            y_train = y
        
        alpha = cho_solve((L, True), y_train)

        log_lik_dims = -.5 * np.einsum("ik, ik->k",y_train, alpha)
        log_lik_dims -= np.log(np.diag(L)).sum()
        log_lik_dims -= K.shape[0] / 2 * np.log(2 * np.pi)
        log_lik = log_lik_dims.sum(-1)  # sum over output dimensions

        if eval_gradient:
            tmp = np.einsum("ik,jk->ijk", alpha, alpha)
            tmp -= cho_solve((L, True), np.eye(K.shape[0]))[:, :, None]

            log_likelihood_gradient_dims = \
                0.5 * np.einsum("ijl,ijk->kl", tmp, K_g_grad)
            log_likelihood_gradient = log_likelihood_gradient_dims.sum(-1)


            log_likelihood_gamma_gradient_dims = \
                0.5 * np.einsum("ijl,ijk->kl", tmp, K_gamma_grad)
            log_likelihood_loggamma_gradient = log_likelihood_gamma_gradient_dims.sum(-1)
            # Gamma parameters is on the log scale
            log_likelihood_loggamma_gradient = \
                log_likelihood_loggamma_gradient * np.exp(loggamma)

            log_likelihood_alpha_gradient_dims = \
                0.5 * np.einsum('ijl,ijk->kl', tmp, K_alpha_grad)
            log_likelihood_logalpha_gradient = log_likelihood_alpha_gradient_dims.sum(-1)
            # alpha parameters are on the log scale
            log_likelihood_logalpha_gradient = \
                log_likelihood_logalpha_gradient * np.exp(logalpha)

            return (log_lik,
                    log_likelihood_gradient,
                    log_likelihood_loggamma_gradient,
                    log_likelihood_logalpha_gradient)
        else:
            return log_lik

    def prior_logpdf(self, g, logpsi, eval_gradient):
        """Logpdf of prior
        """
        veclogpsi = logpsi.copy()
        logpsi_shape = util._get_gp_theta_shape(self.latentforces)
        logpsi = util._unpack_vector(logpsi, logpsi_shape)

        # reshapge g
        g = g.reshape(self.dim.R, self.dim.N)
        ll = 0.

        if eval_gradient:
            ll_g_grad, ll_logpsi_grad = ([], [])

        for r in range(self.dim.R):
            gp = self.latentforces[r]
            kern = gp.kernel.clone_with_theta(logpsi[r])

            if eval_gradient:
                K, K_gradient = kern(self.ttc[:, None],
                                     eval_gradient=True)
            else:
                K = kern(self.ttc[:, None])

            K[np.diag_indices_from(K)] += gp.alpha
            L = np.linalg.cholesky(K)

            alpha = cho_solve((L, True), g[r, :])

            tmp = -.5 * g[r, :].dot(alpha)
            tmp -= np.log(np.diag(L)).sum()

            ll += -.5 * g[r, :].dot(alpha) - \
                  np.log(np.diag(L)).sum() - \
                  L.shape[0] / 2 * np.log(2 * np.pi)

            if eval_gradient:
                ll_g_grad.append(-alpha)

        if eval_gradient:
            return ll, \
                   np.concatenate(ll_g_grad)

        return ll


class VarMLFMSuccApprox(BaseMLFMSuccApprox):

    def varfit(self, times, Y, gtol=1e-3, max_iter=100, **kwargs):

        try:
            mapres = kwargs.pop('mapres')
            self._setup(times, **kwargs)
        except KeyError:
            raise NotImplementedError("varfit currently requires the results of a MAP fit")

        # initalise the distribution of g at the MAP
        Eg = mapres.g.ravel()
        Covg = np.diag([0.1]*Eg.size)

        Eb = mapres.beta.ravel()

        Deltag = np.inf

        # initalise the distribution of X
        # X is of shape[N*K, order+1, n_samples]
        Ex, Covx = self._init_x_distrib(times, Y)
        # Burn in the distribution of Ex, Covx around g map

        for i in range(5):
            Ex, Covx = self.x_update_moments(Y, Ex, Covx, Eg, Covg,
                                             mapres.beta, np.exp(mapres.loggamma),
                                             mapres.logphi, mapres.logtau)
        
        nt = 0  # Count number of iterations
        while nt < 10:
            Eg, Covg = self.g_update_moments(Ex, Covx,
                                             mapres.beta, np.exp(mapres.loggamma),
                                             mapres.logpsi)            
            Ex, Covx = self.x_update_moments(Y,
                                             Ex, Covx,
                                             Eg, Covg,
                                             mapres.beta, np.exp(mapres.loggamma),
                                             mapres.logphi, mapres.logtau)
            nt += 1

        return Eg, Covg

    def _init_x_distrib(self, times, Y):
        from scipy.interpolate import interp1d
        if self.is_tt_aug:
            # cheap interpolation of the data to the latent variable times
            Ex = np.zeros((self.dim.N*self.dim.K, self.order+1, Y.shape[1]))
            for m, Ym in enumerate(Y.T):
                Ym = Ym.reshape(self.Ndata, self.dim.K)
                u = [interp1d(times, ymk,
                              kind='cubic',
                              fill_value='extrapolate')
                     for ymk in Ym.T]

                Yinterp = np.column_stack([uk(self.ttc)
                                           for uk in u]).ravel()
                Ex[:, :, m] = np.column_stack((Yinterp
                                               for p in range(self.order+1)))
            NK = self.dim.N*self.dim.K
            
            Covx = np.dstack([np.diag(0.001*np.ones(NK))
                              for p in range(min(self.order, 3))])
            return Ex, Covx

    def g_update_moments(self, EX, CovX, beta, gamma, logpsi):

        # dimensionality variables
        n_outputs = EX.shape[-1]
        NK = self.dim.N*self.dim.K

        # struct mats
        struct_mats = np.array([sum(brd*Ld for brd, Ld in zip(br, self.basis_mats))
                                for br in beta])

        # vec(K) = v0 + V.dot(g)
        V, v0 = self.sparse_vecK_aff_rep(beta)
        V = V.todense()
        v0 = v0.todense()

        GammaInv = np.diag(np.concatenate([(1/gk)*np.ones(self.dim.N)
                                           for gk in gamma]))

        # array of outer products
        EXXT = np.einsum('ikl,jkl->ijkl', EX[...,:-1,:], EX[...,:-1,:])
        # ... adjust with covariance matrices
        EXXT[..., 0, :] += CovX[..., 0][..., None]
        if EXXT.shape[1] > 2:
            EXXT[..., 1:-1, :] += CovX[..., 1][..., None, None]
            EXXT[..., -1, :] += CovX[..., 2][..., None]
        else:
            EXXT[..., -1, :] += CovX[..., 1]
        S = np.kron(EXXT.sum(2), GammaInv[..., None])

        ## E outer(Xm, X_{m+1})
        # - we use the fact cov{X_m, X_{m+1}} = 0 under chosen mean field factorisation
        EXXpT = np.einsum('ikl,jkl->ijkl', EX[..., 1:, :], EX[..., :-1, :]).sum(2)

        # b = vec(sum([Gamma.dot(outer(xm, xm-1)) ]))
        b = np.einsum('ij,jkl->ikl', GammaInv, EXXpT)
        b = b.transpose(1, 0, 2).reshape(NK*NK, n_outputs)

        premean = np.einsum('ijl,jk->ikl', S, v0)
        premean = b[:, None, :] - premean
        premean = np.einsum('ij,jkl->ikl', V.T, premean)
        
        invcov = np.einsum('ijl,jk->ikl', S, V)
        invcov = np.einsum('ij,jkl->ikl', V.T, invcov)

        invcov = invcov.sum(-1)
        # get the contribution from the prior
        logpsi = util._unpack_vector(logpsi,
                                     util._get_gp_theta_shape(self.latentforces))
        for r, theta in enumerate(logpsi):
            kern = self.latentforces[r].kernel.clone_with_theta(theta)
            K = kern(self.ttc[:, None])
            K[np.diag_indices_from(K)] += self.latentforces[r].alpha
            L = np.linalg.cholesky(K)
            invcov[r*self.dim.N:(r+1)*self.dim.N,
                   r*self.dim.N:(r+1)*self.dim.N] += \
                   cho_solve((L, True), np.eye(L.shape[0]))

        L = np.linalg.cholesky(invcov)
        cov = cho_solve((L, True), np.eye(L.shape[0]))
        mean = cov.dot(premean.sum(-1))
        #mean = np.einsum('ij,jkl->ikl', cov, premean.sum(-1))[:, 0].sum(-1)

        return mean.ravel(), cov

    def x_update_moments(self, Y, EX, CovX, Eg, Covg,
                         beta, gamma, logphi, logtau):

        # Get the additive noise error term
        GammaInv = np.diag(
            np.column_stack([1/gk*np.ones(self.dim.N)
                             for gk in gamma]).ravel())
        
        # x0 cov variance matrix
        logphi = util._unpack_vector(logphi,
                                     util._get_gp_theta_shape(self.latentstates))
        C0inv = [gp.kernel.clone_with_theta(theta)(self.ttc[:, None])
                 for gp, theta in zip(self.latentstates, logphi)]
        for k, C in enumerate(C0inv):
            C[np.diag_indices_from(C)] += self.latentstates[k].alpha
        C0inv = [np.linalg.cholesky(C) for C in C0inv]
        C0inv = block_diag(*[cho_solve((L, True), np.eye(L.shape[0]))
                             for L in C0inv])
        # this is the inverse of Cov{vec(X)}, we need
        # Cov{rav(X)} = T Cov{vec(X)} Tt
        #   ---> InvCov{rav(X)} = Tt InvCov{vec(X)} T
        Tt = util.T_xrav_tovecx(self.dim.N, self.dim.K).transpose()
        C0inv = Tt.dot(Tt.dot(C0inv).T)

        ## Takes the expectation wrt to distribution of G
        EK = self._K(Eg, beta)
        EKt = EK.transpose()
        EKtLK = self._EKtLK(Eg, Covg, beta, GammaInv)

        ######
        # q(X0)
        ic0 = C0inv + EKtLK
        pm0 = EKt.dot(GammaInv.dot(EX[:, 1, :]))
        L0 = np.linalg.cholesky(ic0)
        cov0 = cho_solve((L0, True), np.eye(L0.shape[0]))
        EX[:, 0, :] = cov0.dot(pm0)
        CovX[..., 0] = cov0

        # Common cov
        cov = GammaInv + EKtLK
        L = np.linalg.cholesky(cov)
        cov = cho_solve((L, True), np.eye(L.shape[0]))
        CovX[..., 1] = cov        
        for m in range(self.order-1):
            pm = EKt.dot(GammaInv.dot(EX[:, m+2, :])) + \
                 GammaInv.dot(EK.dot(EX[:, m, :]))
            EX[:, m+1, :] = cov.dot(pm)

        datainvcov = np.diag(
            np.column_stack([tau*np.ones(self.Ndata)
                             for tau in np.exp(logtau)]).ravel())

        datainvcov = [gp.kernel.clone_with_theta(theta)(self._tt_data[:, None])
                 for gp, theta in zip(self.latentstates, logphi)]
        for k, C in enumerate(datainvcov):
            C[np.diag_indices_from(C)] += self.latentstates[k].alpha
        datainvcov = [np.linalg.cholesky(C) for C in datainvcov]
        datainvcov = block_diag(*[cho_solve((L, True), np.eye(L.shape[0]))
                             for L in datainvcov])
        # this is the inverse of Cov{vec(X)}, we need
        # Cov{rav(X)} = T Cov{vec(X)} Tt
        #   ---> InvCov{rav(X)} = Tt InvCov{vec(X)} T
        Tt = util.T_xrav_tovecx(self.Ndata, self.dim.K).transpose()
        datainvcov = Tt.dot(Tt.dot(datainvcov).T)

        invcov = GammaInv
        if self.is_tt_aug:
            data_inds = np.asarray(self.data_inds)
            data_inds = (data_inds*self.dim.K)[:, None] + \
                        np.arange(self.dim.K)[None, :]
            data_inds_ix_ = np.ix_(data_inds.ravel(), data_inds.ravel())
            invcov[data_inds_ix_] += datainvcov
            pm = GammaInv.dot(EK.dot(EX[:, -2, :]))
            pm[data_inds.ravel(), :] += datainvcov.dot(Y)
        else:
            invcov += datainvcov
            pm = datainvcov.dot(Y) + GammaInv.dot(EK.dot(EX[:, -2, :]))

        cov = np.linalg.inv(invcov) #np.diag(1 / np.diag(invcov))
        
        EX[:, -1, :] = cov.dot(pm)
        CovX[..., -1] = cov

        return EX, CovX

    def _EKtLK(self, Eg, Covg, beta, L):
        """
        Computes the expected value of the matrix valued quadratic form.
        """
        EggT = np.outer(Eg, Eg) + Covg

        # vector v0 and matrix V with vec(K) = v0 + V.dot(g)
        V, v0 = self.sparse_vecK_aff_rep(beta)
        EVg = V.dot(Eg[:, None])  # E[ V.dot(g) ] = V.dot(Eg)
        vEVg = v0.dot(EVg.T)      # E[ outer(v, Vg) ] = outer(v, V.dot(Eg))

        # E[outer(vec(K), vec(K)]
        E_vecK_outer = v0.dot(v0.transpose()) + \
                       vEVg + vEVg.T + \
                       V.dot(V.dot(EggT).T)

        # reshape so that [n, m, ...] = E[outer(K[:, n], K[:, m])]
        NK = self.dim.N * self.dim.K
        # numpy matrix to array
        E_vecK_outer = np.asarray(E_vecK_outer)
        E_vecK_outer = E_vecK_outer.reshape(NK, NK, NK, NK).transpose(0, 2, 1, 3)
        # reshape so that [n, m, ...] = vec(E[outer(K[:, m], K[:, m])])
        E_vecK_outer = E_vecK_outer.transpose(0, 1, 3, 2).reshape(NK, NK, NK**2)

        # final results is [n, m] = Tr(E_vecK_outer[n, m, ...].dot(L))
        #   i) vectorise L
        #  ii) multiply and sum for dot product
        return (E_vecK_outer * L.T.ravel()[None, None, :]).sum(-1)

"""
Methods for trapezoidal integration
"""
def get_weights(tt):
    tt = np.asarray(tt)
    W = np.zeros((tt.size, tt.size))
    h = np.diff(tt)
    for i in range(tt.size):
        W[i, :i] += .5*h[:i]
        W[i, 1:i+1] += .5*h[:i]
    return W

def weight_matrix(tt, ifix):

    ttb = tt[:ifix+1]
    ttf = tt[ifix:]

    Wb = get_weights([t for t in reversed(ttb)])
    Wb = np.flip(Wb, (0, 1))
    Wf = get_weights(ttf)

    W = np.zeros((tt.size, tt.size))
    W[:ifix+1, :ifix+1] += Wb
    W[ifix:, ifix:] += Wf

    return W
