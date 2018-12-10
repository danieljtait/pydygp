import numpy as np
from . import util
from .mlfm import BaseMLFM, Dimensions
from scipy.linalg import block_diag, cho_solve
from scipy.stats import norm
from scipy.optimize import minimize
import scipy.sparse as sparse

def _unpack_vector(x, shape):
    res = []
    ntot = 0
    for n in shape:
        res.append(x[ntot:ntot+n])
        ntot += n
    return res

def Mpower_action_wgrad(M, dM, x, power, dMdb=None):
    """
    Returns the result of M^{power}x and its gradient. 
    """
    res = x
    grad = [np.zeros(x.size).reshape(x.shape)]*dM.shape[0]

    if dMdb is not None:
        grad_b = [np.zeros(x.size).reshape(x.shape)]*dMdb.shape[0]
    
    for i in range(power):
        grad = [dM_i.dot(res) + M.dot(grad_i)
                for dM_i, grad_i in zip(dM, grad)]
        if dMdb is not None:
            grad_b = [dM_r.dot(res) + M.dot(grad_i)
                      for dM_i, grad_i in zip(dMdb, grad_b)]
            
        res = M.dot(res)

    if dMdb is None:
        return res, np.array(grad)
    else:
        return res, grad, grad_b

def M2():

    # make K
    
    res = x
    grad = np.empty((N_repl, N, R, NK))
    for i in range(power):
        res, dres = KAction_grad()
        K_dot_grad = np.einsum('ij,mnrj->mnri', W, grad)
        grad += dres + K.dot(grad)
    return res, grad
        

def KAction_grad(x, g, beta, W,
                 N_repl, N, K, R):

    # X[m, k, ...] = kth dimension component of x
    X = x.T.reshape((N_repl, K, N))
    X = X.transpose((0, 2, 1))

    Arr = np.dstack([sum(brd*Ld
                         for brd, Ld in zip(br, self.basis_mats)).T
                     for br in beta[1:, :]])

    # XA[n, ..., r] = X[n, ...].dot(Arr[..., r])
    XA = np.einsum('ijk,klr->ijlk', X, Arr)

    # store the outer product of the nth col. W
    # with the nth row X[, n, ]

    # broadcasting:
    #    W        XA
    #  (N, N)   (M, N, K, R)
    #
    # WXA[m, n, :, k, r] = outer prod of the nth col. of W with
    # the mth replicate of X
    WXA = (W.T)[None, :, :, None, None] * XA[..., None, :, :]

    # now use the gradient to also construct K[g].dot(x) = K[0].dot(x) + sum( g * dKdg )

    # finally transpose and the reshape so that X has been vectorised
    return WXA.reshape((N_repl, N, R, N*K))

def Kpow_Action_grad(x, beta, power, W,
                     N_repl, N, K, R):
    res = x


# Utility functions for padding the time vector
def augment_times(t, h):
    """
    Pads the vector t so that the maximum spacing is h.

    Parameters
    ----------

    t : one dimensional array like

    h : float
    """
    inds = [0]
    res = [t[0]]
    for ta, tb in zip(t[:-1], t[1:]):
        N = np.ceil((tb - ta) / h + 1)
        N = int(N)
        _tt = np.linspace(ta, tb, N)
        res = np.concatenate((res, _tt[1:]))
        inds.append(inds[-1] + N - 1)
    return res, np.array(inds)    

def handle_time_inds(times, h=None):
    """
    Takes a list of time vectors and returns the unique, potentially augmented,
    vector 
    """
    # get size of each vector
    t_sizes = [len(t) for t in times]

    # concatenate to single time vector
    tt = np.concatenate(times)

    # get the distinct times, and the indices
    tt_uni, inv_ind = np.unique(tt, return_inverse=True)

    # split inv_ind up
    ind_ti = _unpack_vector(inv_ind, t_sizes)

    if h is None:
        return tt_uni, ind_ti

    elif isinstance(h, float) and h > 0:
        # augment the time vector so that diff is at most h
        ttc, inds_c = augment_times(tt_uni, h)
        data_inds = [inds_c[ind_i] for ind_i in ind_ti]
        return ttc, data_inds

    else:
        raise ValueError("h should be a float > 0")

class Softmax:
    def __init__(self, basis_funcs):
        self.basis_funcs = basis_funcs
        self.param_shape = [len(funcs) for funcs in basis_funcs]

    def __call__(self, v, X):
        if len(v) != sum(self.param_shape):
            raise ValueError("v should be an array like of"
                             " shape {}".format(sum(self.param_shape)))
        else:
            v = util._unpack_vector(v, self.param_shape)

        F = [np.column_stack((f(X) for f in bfuncs))
             for bfuncs in self.basis_funcs]

        exp_activs = np.column_stack(
            [ (Fi*vi).sum(-1) for Fi, vi in zip(F, v)])
        exp_activs = np.exp(exp_activs)

        return exp_activs / exp_activs.sum(-1)[:, None]

class EMFitMixin:
    """
    Handles MAP fitting of MLFMSA-MoE model using the EM algorithm.
    """
    def _setup_softmax(self, ifix, basis_funcs):

        assert(len(ifix) == len(basis_funcs))
        tfix = [self.ttc[i] for i in ifix]

        def wrapper(foo, ti):
            def f(x):
                return foo(x, ti)
            return f

        # wrap all the functions to demove the denese on fi
        centered_basis_funcs = []
        for bf, ti in zip(basis_funcs, tfix):
            new_bf = []
            for func in bf:
                new_bf.append(wrapper(func, ti))
            centered_basis_funcs.append(new_bf)

        self.softmax_activs = Softmax(centered_basis_funcs)

    def _get_responsibilities(self, pi, g, beta, mu_ivp, alpha_prec, ifix):
        """
        Returns
        -------
            responsibilities: [r1, ..., rQ]
        """
        probs = [[]]*len(self.N_data)  # store log probs for each replicate
        for i, ifx in enumerate(ifix):
            K = self._K(g, beta, ifx)
            Lop = np.linalg.matrix_power(K, self.order)

            means = np.kron(mu_ivp[:, i, :],
                            np.ones(self.dim.N)).T
            means = Lop.dot(means)
            
            for q, replicate in enumerate(self.y_train_):
                # subsample mean using data inds
                m_i_q = means[:, q].reshape(self.dim.K, self.dim.N).T
                m_i_q = m_i_q[self.data_inds[q], :]
                lp_i_q = norm.logpdf(
                    replicate.reshape(self.dim.K, self.N_data[q]).T,
                    loc=m_i_q,
                    scale=1/np.sqrt(alpha_prec))

                # sum over the dimesnions
                lp_i_q = lp_i_q.sum(-1)

                if probs[q] == []:
                    probs[q] = lp_i_q[:, None]
                else:
                    probs[q] = np.column_stack((probs[q], lp_i_q))

        probs = [np.exp(lp)*pi_p for lp, pi_p in zip(probs, pi)]
        sum_probs = [item.sum(-1) for item in probs]

        return [p / sum_p[:, None] for p, sum_p in zip(probs, sum_probs)]

    def _optim_g(self, g, beta, mu_ivp,
                 alpha_prec, responsibilities,
                 ifix, gprior):

        resp = [np.row_stack([r_q]*self.dim.K)
                for r_q in responsibilities]

        # inverse covariance of g
        Cg = block_diag(*[gp.kernel(self.ttc[:, None])
                          for gp in self.latentforces])
        Cg[np.diag_indices_from(Cg)] += 1e-5
        Lg = np.linalg.cholesky(Cg)

        # going to optimise mu_ivp too
        # shape (N_replicates, N_ifix, ...) they are all independent
        ones = np.ones(self.dim.N)

        # Useful dimensions
        N_repl = len(self.y_train_)
        N_mixcomp = len(ifix)        
        
        inds = [np.concatenate([self.data_inds[m] + k*self.dim.N
                                for k in range(self.dim.K)])
                for m in range(len(self.y_train_))]

        def _g_objfunc(arg):
            g = arg[:self.dim.N*self.dim.R]
            #_mu_ivp = arg[self.dim.N*self.dim.R:]
            #mu_ivp = _mu_ivp.reshape((N_repl, N_mixcomp, self.dim.K))

            log_lik, log_lik_g_grad = (0, np.zeros(g.size))
            log_lik_mu_ivp_grad = np.empty((mu_ivp.shape))

            for i, ifx in enumerate(ifix):
                # sum over the mixture components first so we don't have to
                # recalculate the gradient of the matrix operator
                Ki, Ki_grad = self._K(g, beta, ifx, eval_gradient=True)

                # ivp_means is shape [N_replic., N_ifix, self.dim.K]
                mu_ivp_ifx = mu_ivp[:, i, :]
                # means will be shape (NK, N_replicates)
                means = np.kron(mu_ivp[:, i, :],
                                np.ones(self.dim.N)).T

                # raise Ki to the correct order and get its gradient

                # is this redudnant?
                #Li = np.linalg.matrix_power(Ki, self.order)
                
                Lim, Lim_grad = Mpower_action_wgrad(Ki, Ki_grad, means, self.order)

                # sum over the replicates
                for q, replicate in enumerate(self.y_train_):
                    # sub sample the means to match up with the time
                    # indices of replicate q
                    # note data_inds[q] is an np.array
                    inds_q = inds[q]
                    Lim_q = Lim[inds_q, q]

                    eta = Lim_q - replicate
                    r_q_i = resp[q][:, i]  # responsibilities for the qth replicate
                                           # of the ith mixture component
                    Reta = eta * r_q_i     # weight eta by the responsibilites

                    # update the log-likelihood and gradient
                    log_lik += -0.5 * (eta * Reta).sum() * alpha_prec
                    log_lik_g_grad -= alpha_prec * np.array(
                        [np.sum(Reta * dL[inds_q, q]) for dL in Lim_grad])

                    # gradient with respect to the ivp mean
                    """
                    ei = np.zeros(self.dim.K)
                    for k_dim in range(self.dim.K):
                        ei[k_dim] = 1.
                        deta = Li.dot(np.kron(ei, ones))[inds_q]
                        log_lik_mu_ivp_grad[q, i, k_dim] = \
                                               - alpha_prec * deta.dot(Reta)
                        ei[k_dim] = 0.
                    """

                # add the contribution from the prior
                alpha = cho_solve((Lg, True), g)
                log_lik -= .5 * g.dot(alpha)
                log_lik_g_grad -= alpha

            grad = np.concatenate((log_lik_g_grad,))
            #log_lik_mu_ivp_grad.flatten()))
            return -log_lik, -grad

        # flatten _mu_ivp (flattens along rows
        _mu_ivp = mu_ivp.reshape(N_repl*N_mixcomp*self.dim.K)

        x0 = np.concatenate((g,))# _mu_ivp))

        # optimize
        res = minimize(_g_objfunc, x0, jac=True, tol=1e-3)
        # unpack the results
        g = res.x[:self.dim.R*self.dim.N]
        #mu_ivp = res.x[self.dim.R*self.dim.N:]
        #mu_ivp = mu_ivp.reshape(N_repl, N_mixcomp, self.dim.K)

        return g, mu_ivp

    def _optim_pi_par(self, v, responsibilities, vprior=None):
        """
        EM optimisation of the parameters of the softmax activations.
        """

        def _v_objfunc(v):
            val, grad = (0, np.zeros(len(v)))

            for q, x_q in enumerate(self.x_train_):

                pi = self.softmax_activs(v, x_q[:, None])

                val += np.sum(responsibilities[q] * np.log(pi))

                # calculate the gradient
                F = [np.column_stack((f(x_q[:, None])
                                      for f in bfuncs))
                     for bfuncs in self.softmax_activs.basis_funcs]
                
                _grad = []
                for i, Fi in enumerate(F):
                    d_i = responsibilities[q][:, i][:, None] * Fi
                    d_i -= Fi * pi[:, i][:, None]
                    _grad.append(d_i.sum(0))
                _grad = np.concatenate(_grad)
                grad += np.array(_grad)

            if vprior is not None:
                lp, lp_grad = vprior.logpd(v, eval_gradient=True)
                val += lp
                grad += lp_grad

            return -val, -grad

        
        res = minimize(_v_objfunc, x0=v, jac=True)
        return res.x

    def _initialse_Z(self):
        from scipy.interpolate import interp1d
        zz = []
        for t, y in zip(self.x_train_, self.y_train_):
            y = y.reshape(self.dim.K, t.size).T
            u = [interp1d(t, yk, kind='cubic', fill_value='extrapolate')
                 for yk in y.T]
            z = np.concatenate([uk(self.ttc) for uk in u])
            zz.append(z)

        # construct the statistics sum[zzT] sum[z_i z_{i+1}T]
        S0 = [self.order*np.outer(z, z) for z in zz]
        S1 = [self.order*np.outer(z, z) for z in zz]
        return S0, S1

    def _update_z_dist2(self, g, beta, ifx, lam, alf, mu_ivp):
        gp = self.latentforces[0]
        Cz = [gp.kernel(self.ttc[:, None])]*self.dim.K
        Lz = []
        for c in Cz:
            c[np.diag_indices_from(c)] += 1e-5
            Lz.append(np.linalg.cholesky(c))

        Cz_inv = block_diag(*[cho_solve((L, True),
                                        np.eye(L.shape[0]))
                              for L in Lz])

        K = self._K(g, beta, ifx)

        # parameters for the LDS update
        q = 0
        Sigma = np.eye(self.N_data[q]*self.dim.K) / alf

        y = self.y_train_[q]

        Gamma_inv = np.eye(self.dim.N*self.dim.K) * alf # + Cz_inv / lam


        A = K #Gamma.dot(K)
        C = np.zeros((self.N_data[q]*self.dim.K,
                      self.dim.N*self.dim.K))

        inds = self.data_inds[q]
        inds = np.concatenate([inds + self.dim.N*k
                               for k in range(self.dim.K)])
        for i in range(C.shape[0]):
            C[i, inds[i]] += 1

        Cz = block_diag(*Cz)
        Sigma = 0.01*C.dot(Cz.dot(C.T))
        Gamma = np.eye(self.dim.N*self.dim.K) / alf
        #Gamma = 0.01*np.linalg.inv(Cz_inv) #np.linalg.inv(Gamma_inv)        


        u1 = np.kron(mu_ivp[0, 0, :], np.ones(self.dim.N))
        u1 = np.zeros(self.dim.N*self.dim.K)
        V1 = np.ones((self.dim.N*self.dim.K,self.dim.N*self.dim.K))
        V1 = 100.*np.eye(V1.shape[0])

        P1 = A.dot(V1.dot(A.T)) + Gamma
        K2 = P1.dot(C.T.dot(np.linalg.inv(C.dot(P1.dot(C.T)) + Sigma)))
        u2 = A.dot(u1) + K2.dot(y - C.dot(A.dot(u1)))
        V2 = (np.eye(K2.shape[0]) - K2.dot(C)).dot(P1)

        J1 = V1.dot(A.T.dot(np.linalg.inv(P1)))

        u1h = u1 + J1.dot(u2 - A.dot(u1))
        V1h = V1 + J1.dot(V2 - P1).dot(J1.T)

        means = (u1h, u2)
        covs = (V1h, V2)
        pwcovs = (J1.dot(V2), )
        return means, covs, pwcovs
                                

    def _update_z_dist(self, g, beta, ifx, lam=1000):

        # prior inv. covariance matrix for z
        gp = self.latentforces[0]
        Cz = [gp.kernel(self.ttc[:, None])]*self.dim.K
        Lz = []
        for c in Cz:
            c[np.diag_indices_from(c)] += 1e-5
            Lz.append(np.linalg.cholesky(c))

        Cz_inv = block_diag(*[cho_solve((L, True),
                                        np.eye(L.shape[0]))
                              for L in Lz])

        # construct the data indices transform
        q = 0
        inds = self.data_inds[q]
        inds = np.concatenate([inds + self.dim.N*k
                               for k in range(self.dim.K)])

        C = np.zeros((self.N_data[q]*self.dim.K,
                      self.dim.N*self.dim.K))

        for i in range(C.shape[0]):
            C[i, inds[i]] += 1


        K = self._K(g, beta, ifx)
        CK = C.dot(K)

        A = C.dot(K)
        
        invcov = Cz_inv + lam * A.T.dot(A)
        premean = A.T.dot(self.y_train_[q]) * lam

        #dot(K)).T.dot(self.y_train_[q])
        cov = np.linalg.inv(invcov)
        mean = cov.dot(premean) 

        return mean, cov, K.dot(mean)

    def _update_g(self, Ez, EzzT, beta, ifx, lam):
        q = 0
        inds = self.data_inds[q]
        inds = np.concatenate([inds + self.dim.N*k
                               for k in range(self.dim.K)])
        C = np.zeros((self.N_data[q]*self.dim.K,
                      self.dim.N*self.dim.K))

        for i in range(C.shape[0]):
            C[i, inds[i]] += 1
        CTC = C.T.dot(C)

        Cg = [gp.kernel(self.ttc[:, None])
              for gp in self.latentforces]
        for c in Cg:
            c[np.diag_indices_from(c)] += 1e-5
        Lg = [np.linalg.cholesky(c) for c in Cg]

        invcov = block_diag(*[
            cho_solve((L, True), np.eye(self.dim.N*self.dim.R))
            for L in Lg])

        V, v = self._vecK_aff_rep(beta, ifx)
        S0 = np.kron(EzzT, CTC)
        invcov += lam * V.T.dot(S0.dot(V))
        premean = np.outer(C.T.dot(self.y_train_[q]), Ez).T.ravel()
        premean -= v.dot(S0)
        premean = lam * premean.dot(V)

        return np.linalg.solve(invcov, premean)

        
    def _EM_g_update(self, S0, S1, beta, ifx):
        lam = 1e5
        # inverse covariance of g
        Cg = [gp.kernel(self.ttc[:, None])
              for gp in self.latentforces]
        for c in Cg:
            c[np.diag_indices_from(c)] += 1e-5
        Lg = [np.linalg.cholesky(c) for c in Cg]

        invcov = block_diag(*[
            cho_solve((L, True), np.eye(self.dim.N*self.dim.R))
            for L in Lg])

        S_x_I = np.kron(S0, np.eye(self.dim.N*self.dim.K))

        # prior inv covariance
        V, v = self._vecK_aff_rep(beta, ifx)
        invcov += lam*V.T.dot(S_x_I.dot(V))

        # 'pre-mean'
        premean = S1.T.ravel() - v.dot(S_x_I)
        premean = lam*premean.dot(V)

        return np.linalg.solve(invcov, premean)


    def _Kalman_g_update(self, g, beta, mu_ivp, ifix, alpha_prec):
        for i, ifx in enumerate(ifix):

            lam = 1e-5
            Gamma = lam*np.diag(np.ones(self.dim.N*self.dim.K))
            Sigma = np.diag(np.ones(self.dim.N*self.dim.K) / alpha_prec )

            Ki = self._K(g, beta, ifx)
            mu_ivp_ifx = mu_ivp[:, i, :]

            mu0 = np.kron(mu_ivp[:, i, :],
                          np.ones(self.dim.N)).T

            muf, Vf = ([], [])  # mean and covar. for the formward sweep of
                                # the Kalman filter

            #P0 = Gamma#1e-4*np.eye(self.dim.N*self.dim.K)
            P0 = lam*np.ones((self.dim.N*self.dim.K,
                            self.dim.N*self.dim.K))
            Pchol = []

            ## Forwrad sweep
            for nt in range(self.order-1):
                if nt == 0:
                    muf.append(mu0)
                    Vf.append(P0)
                else:
                    muf.append(Ki.dot(muf[-1]))
                    Vf.append(Ki.dot(Ki.dot(Vf[-1]).T))

                P = Ki.dot(Ki.dot(Vf[-1]).T) + Gamma
                P[np.diag_indices_from(P)] += 1e-6
                Pchol.append(np.linalg.cholesky(P))

            # Now update final term in the forward sweep which contains
            # the data
            _m = Ki.dot(muf[-1])
            mfinal = _m.copy()

            for q, inds in enumerate(self.data_inds):

                inds = np.concatenate([inds + self.dim.N*k
                                       for k in range(self.dim.K)])
                C = np.zeros((self.dim.N*self.dim.K, self.N_data[q]*self.dim.K))
                for i in range(C.shape[1]):
                    C[inds[i], i] += 1
                C=C.T

                Sigma = np.diag(np.ones(self.N_data[q]*self.dim.K) / alpha_prec )

                gp = self.latentforces[0]
                #Sigma = lam*block_diag(*[gp.kernel(self.x_train_[q][:, None])
                #                         for k in range(self.dim.K)])
                
                inds_ix = np.ix_(inds, inds)
                #CPCt = P[inds_ix]  # subsample for where we have data
                CPCt = P[inds_ix]

                # shape (Naug, Ndat_q)
                Kgain = P.dot(C.T.dot(np.linalg.inv(CPCt + Sigma)))

                mfinal[:, q] += Kgain.dot(self.y_train_[q] - C.dot(_m[:, q]))

                #Vfinal = (np.eye(self.N_data[q]*self.dim.K) - \
                #          Kgain[inds, :]).dot(P[inds, :])
                Vfinal = (np.eye(Kgain.shape[0]) - Kgain.dot(C)).dot(P)

                muf.append(mfinal)
                Vf.append(Vfinal)
            # Backward sweep
            pw_covs = []
            KiT = Ki.T

            for nt in range(self.order-1):

                Jn = Vf[-(nt+2)].dot(
                    KiT.dot(
                    cho_solve((Pchol[-(nt+1)], True),
                              np.eye(self.dim.N*self.dim.K))))

                mn = muf[-(nt+2)] + \
                     Jn.dot(muf[-(nt+1)] - Ki.dot(muf[-(nt+2)]))

                Pn = Pchol[-(nt+1)].dot(Pchol[-(nt+1)].T)
                Vn = Vf[-(nt+2)] - \
                     Jn.dot(Jn.dot(Vf[-(nt+1)] - Pn).T)
                muf[-(nt+2)] = mn
                Vf[-(nt+2)] = Vn

                pw_covs.append(Jn.dot(Vf[-(nt+1)]))

            # get the two statistics of interest
            S0 = 0.
            for V, m in zip(Vf[:-1], muf[:-1]):
                S0 += V + np.outer(m[:, 0], m[:, 0])
            S1 = 0.
            for i, pwV in enumerate(pw_covs):
                S1 += pw_covs[0] + np.outer(muf[i][:, 0], muf[i+1][:, 0])

            return S0, S1, muf, Vfinal

            # repr. of Ki = v0 + V.dot(g)
            V, v0 = self._vecK_aff_rep(beta, ifx)

            S0 = np.kron(S0, np.eye(self.dim.N*self.dim.K))

            expr1 = V.T.dot(S0.dot(V))
            expr2 = S1.T.ravel() - v0[None, :].dot(S0)
            expr2 = expr2.dot(V)

            # inverse covariance of g
            Cg = [gp.kernel(self.ttc[:, None])
                  for gp in self.latentforces]
            for c in Cg:
                c[np.diag_indices_from(c)] += 1e-5
            Lg = [np.linalg.cholesky(c) for c in Cg]

            inv_cov = block_diag(*[
                cho_solve((L, True), np.eye(self.dim.N*self.dim.R))
                for L in Lg])
            #inv_cov *= lam
            #inv_cov += expr1
            c = np.linalg.inv(inv_cov)


            mean = np.linalg.solve(inv_cov, expr2[0, :])
            
            z0 = muf[-2][:, 0]
            z1 = muf[-1][:, 0]

            ExxT = Vf[-2] + np.outer(z0, z0)

            A = np.kron(z0[None, :], np.eye(self.dim.N*self.dim.K))
            A = A.dot(V)
            b = np.kron(z0[None, :], np.eye(self.dim.N*self.dim.K)).dot(v0)

            #A += np.row_stack([inv_cov]*self.dim.K)
            
            mean = np.linalg.lstsq(A, z1 - b, rcond=None)
            rank = mean[2]
            mean = mean[0]

            ic = V.T.dot(np.kron(np.outer(z0, z0),
                                 np.eye(self.dim.N*self.dim.K))).dot(V)
            ic += inv_cov * lam
            pm = np.kron(z0[:, None], np.eye(self.dim.N*self.dim.K)).dot(z1[:, None])
            pm -= np.kron(np.outer(z0, z0), np.eye(self.dim.N*self.dim.K)).dot(v0[:, None])
            pm = V.T.dot(pm)
#            pm /= lam

            mean = np.linalg.solve(ic, pm)
            
            return muf[-1], mean



class BaseMLFMSA(BaseMLFM):

    is_tt_aug = False  # allows the possibility to augment the time vector

    def __init__(self, *args, order=1, **kwargs):
        super(BaseMLFMSA, self).__init__(*args, **kwargs)

        if isinstance(order, int) and order > 0:
            self.order = order

        else:
            raise ValueError("The approximation error must be a"
                             " positive integer.")

    def _setup_times(self, tt, tt_aug=None,
                     h=None, multi_output=False):

        if not multi_output:
                tt = [tt, ]

        # numpy arrayify time vectors
        tt = [np.asarray(item) for item in tt]
        
        if True:# h is not None:
            # h is checked first
            ttc, data_inds = handle_time_inds(tt, h)
            self.x_train_ = tt
            self.ttc = ttc
            self.data_inds = data_inds
            self.is_tt_aug = True

        elif tt_aug is not None:
            # User has supplied the time points at which they want
            # to estimate the latent forces

            if multi_output:
                # add tt_aug to the times list
                tt = [tt_aug] + [t for t in tt]
                ttc, data_inds = handle_time_inds(tt, h)
                self.x_train_ = tt[1:]
                self.ttc = ttc
                self.data_inds = data_inds[1:]
                self.is_tt_aug = True

        else:
            # neither h or tt_aug is supplied
            self.x_train_ = tt
            self.ttc = ttc
            self.is_tt_aug = None

        # store the dimension variables
        _, K, R, D = self.dim
        self.dim = Dimensions(self.ttc.size, K, R, D)
        self.N_data = tuple(item.size for item in self.x_train_)

    def _get_weight_matrix(self, tt, i):
        """
        Weight matrix for the trapezoidal rule.
        """

        def _get_weights(tt):
            tt = np.asarray(tt)
            W = np.zeros((tt.size, tt.size))
            h = np.diff(tt)
            for i in range(len(tt)):
                W[i, :i] += .5 * h[:i]
                W[i, 1:i+1] += .5 * h[:i]
            return W

        ttb = tt[:i+1]
        ttf = tt[i:]

        Wb = _get_weights([t for t in reversed(ttb)])
        Wb = np.flip(Wb, (0, 1))
        Wf = _get_weights(ttf)

        W = np.zeros((len(tt), len(tt)))
        W[:i+1, :i+1] += Wb
        W[i:, i:] += Wf

        return W

    def _K(self, g, beta, ifix, eval_gradient=False, eval_b_gradient=False):
        """
        Return the discretisation of the integral opearator in Picard iteration.
        """
        W = self._get_weight_matrix(self.ttc, ifix)
        g = g.reshape((self.dim.R, self.dim.N)).T
        g = np.column_stack((np.ones(self.dim.N), g))
        
        # struct matrices A_r = sum_r g_rn * sum_d brd * Ld
        struct_mats = np.array([sum(brd*Ld
                                    for brd, Ld in zip(br, self.basis_mats))
                                for br in beta])
        At = struct_mats[None, ...] * g[..., None, None]
        At = At.sum(1)  # sum over R
        
        eifx = np.eye(1, self.dim.N, ifix)
        
        K = sum(np.kron(At[i, ...],
                        np.kron(W[:, i][:, None], np.eye(1, self.dim.N, i)))
                for i in range(self.dim.N))

        I = np.eye(self.dim.K)
        
        K += sum(np.kron(I, np.kron(np.eye(1, self.dim.N, i).T, eifx))
                 for i in range(self.dim.N))

        if eval_gradient:
            K_grad = np.array([np.kron(struct_mats[r+1],
                                       np.kron(W[:, i][:, None],
                                               np.eye(1, self.dim.N, i)))
                                   for r in range(self.dim.R)
                               for i in range(self.dim.N)])
            if eval_b_gradient:
                # This is lazy
                dB = self._dKdB(g, beta, ifix)
                return K, K_grad, dB

            return K, K_grad
        else:
            return K

    def _dKdB(self, g, beta, ifix):
        """
        Rewriting of this function to reduce redundant computations
        """
        W = self._get_weight_matrix(self.ttc, ifix)
        

        # for computational efficiency we need to do
        # any scalar multiplications *before* any
        # kronecker products
        GW = np.stack((gn[:, None] * W[:, n][None, :]
                       for n, gn in enumerate(g)))

        dKdB = []
        for r in range(self.dim.R+1):
            for d in range(self.dim.D):
                dKdB.append(np.kron(self.basis_mats[d],
                                    GW[:, r, :].T))

        return dKdB



    def _Kx_power(self, x,
                  g, beta, power, ifx,
                  eval_g_grad=False, eval_b_grad=False):
        pass
        #if eval_g_grad and eval_b_grad:
        #    K, dKdg, dKdb = self._K(...)


    

    def _Kx(self, G, beta, x, ifx):
        """
        Returns the action of K on a vector x
        """
        xshape = x.shape         # K acts on the final dimension of x
        naxes = len(x.shape) - 1 # number of axes to ignore
        if x.shape[-1] != self.dim.N*self.dim.K:
            raise ValueError("x[..., -1] must be of size {}{}".format(self.dim.K, self.dim.N))

        X = x.reshape(xshape[:-1] + (self.dim.K, self.dim.N))
        X = X.transpose(tuple(range(naxes)) + (naxes+1, naxes))

        # XL[..., N, D, ...] = X[..., :, :].dot(L[d, ...])
        XL = np.einsum('...jk,...dlk->...djl', X, self.basis_mats)

        # get the quadrature weight matrix
        W = self._get_weight_matrix(self.ttc, ifx)

        # WXL shape (..., D, N, N, K)
        # WXL[..., d, n, ...] gives the outer product of the nth
        # col of W with the the nth row of XL[d, ...]
        WXL = (W.T)[..., None] * XL[..., None, :]
        
        # transpose and then flatten so that last component of array
        # is a vectorisation
        nshape = len(WXL.shape)
        perm = tuple(range(nshape-2)) + (nshape-1, nshape-2)
        WXL = WXL.transpose(perm)
        WXL = WXL.reshape(WXL.shape[:-2] + (self.dim.K*self.dim.N, ))

        # Gradient of action with respect to latent forces
        #  - multiply be beta and then sum over d=1,..., D
        #dG = beta[..., None, None] * WXL[..., None, :, :, :]
        #dG = dG.sum(2)

        # Gradient of action with respect to beta
        #  - multiply by G and then sum over n=1,...,N

        # pad G with col of ones
        Gt = G.T

        
        dB = Gt[:, None, :, None] * WXL[..., None, :, :, :]
        dB = dB.sum(3)

        # typically dB is smaller so go for
        Kx = beta[..., None] * dB[..., :, :, :]
        Kx = Kx.sum((1, 2))
        
        # Get the fixed value 
        fixedval = x[..., ifx::self.dim.N]
        xfix = np.kron(fixedval, np.ones(self.dim.N))

        Kx += xfix

        return Kx#, dG, dB

    def _Kx_power(self, g, beta, ifx, x, power):
        # stored a single copy of K
        K = self._K(g, beta, ifx)
        G = np.column_stack((np.ones(self.dim.N),
                             g.reshape(self.dim.R, self.dim.N).T))

        res = x

        gradshape = x.shape[:-1] + (self.dim.R, self.dim.N, x.shape[-1])
        grad = np.empty(gradshape)

        for m in range(power):
            # get K^m x, dK K^{m-1}x
            #res, dKm, _ = self._Kx(G, beta, res, ifx)
            res = self._Kx(G, beta, res, ifx)

            # get K.dot( d [K^{m-1}] )
            #KdKmprev = np.einsum('ij,...nrj->...nri', K, grad)

            # update grad
            #grad = dKm[..., 1:, :, :] + KdKmprev

        return res, grad

    def _vecK_aff_rep(self, beta, ifix, eval_gradient=False):

        struct_mats = np.array([sum(brd*Ld
                                    for brd, Ld in zip(br, self.basis_mats))
                                for br in beta])

        W = self._get_weight_matrix(self.ttc, ifix)

        I = np.eye(self.dim.K)

        NK = self.dim.N*self.dim.K
        K = self.dim.K
        cols = []
        for r in range(self.dim.R):
            for n in range(self.dim.N):
                expr = np.kron(struct_mats[r+1],
                               np.kron(W[:, n][:, None],
                                       np.eye(1, self.dim.N, n)))
                cols.append(expr.T.ravel())
        V = np.column_stack(cols)
        v0 = self._K(np.zeros(self.dim.N*self.dim.R), beta, ifix).T.ravel()
        return V, v0
        
    def bar(self, g, beta, mu_ivp,
            alpha_prec, ifix):
        x = np.kron(mu_ivp[:, 0, :],
                    np.ones(self.dim.N)).T
        x = x[:, 0]
        K = self._K(g, beta, 0)

        x1 = K.dot(x)
        x2 = K.dot(x1)
        x3 = K.dot(x2)



        ########
        ifx = 0
        z0 = x2
        z1 = x3
        lam = 1e5

        # repr. of Ki = v0 + V.dot(g)
        V, v0 = self._vecK_aff_rep(beta, ifx)        


        A = np.kron(z0[None, :], np.eye(self.dim.N*self.dim.K))
        A = A.dot(V)
        b = np.kron(z0[None, :], np.eye(self.dim.N*self.dim.K)).dot(v0)

        #A += np.row_stack([inv_cov]*self.dim.K)
            
        mean = np.linalg.lstsq(A, z1 - b, rcond=None)
        rank = mean[2]
        mean = mean[0]
        

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(self.ttc, g, 'C0-')
        ax.plot(self.ttc, mean, 'r+')
        ax.set_ylim((-2, 2))
        plt.show()

    def fooz(self, g, beta, mu_ivp,
             alpha_prec, responsibilites, gtrue):
        """
        Constrcts an estimate of the distribution of Z
        """
        mean, cov, Kmean = self._update_z_dist(g, beta, 0, lam=10000)
        
        import matplotlib.pyplot as plt
        mean = mean.reshape(self.dim.K, self.dim.N).T
        Kmean = Kmean.reshape(self.dim.K, self.dim.N).T

        _m, _covs, _pwcovs = self._update_z_dist2(g, beta, 0, 1e4, 1e4, mu_ivp)

        _mean = _m[0].reshape(self.dim.K, self.dim.N).T
        _Kmean = _m[1].reshape(self.dim.K, self.dim.N).T

        S0 = np.outer(_m[0], _m[0]) + _covs[0]
        S1 = np.outer(_m[0], _m[1]) + _pwcovs[0]

        """
        fig, ax = plt.subplots()
        
        ax.plot(self.ttc, mean, 'r-')
        ax.plot(self.ttc,
                Kmean, 'o')
        ax.plot(self.x_train_[0],
                self.y_train_[0].reshape(self.dim.K, self.N_data[0]).T, 'C0--', alpha=0.4)
        """

        fig, ax = plt.subplots()
        ax.plot(self.ttc, _mean, 'b-')
        ax.plot(self.ttc, _Kmean, 'r-')
        ax.plot(self.x_train_[0],
                self.y_train_[0].reshape(self.dim.K, self.N_data[0]).T, 'C0--', alpha=0.4)

        ifx = 0
        #z0 = mean.T.ravel()
        z0 = _m[0]
        z1 = _m[1]
        #z1 = Kmean.T.ravel()
        lam = 1e5

        # repr. of Ki = v0 + V.dot(g)
        V, v0 = self._vecK_aff_rep(beta, ifx)        

        A = np.kron(z0[None, :], np.eye(self.dim.N*self.dim.K))
        A = A.dot(V)
        b = np.kron(z0[None, :], np.eye(self.dim.N*self.dim.K)).dot(v0)

        #S0 = np.outer(z0, z0) + cov
        #S1 = np.outer(z0, z1) + 1e-5*np.eye(z1.size)
        _A = V.T.dot(np.kron(S0, np.eye(self.dim.N*self.dim.K)).dot(V))
        
        Cg = [gp.kernel(self.ttc[:, None])
              for gp in self.latentforces]
        for c in Cg:
            c[np.diag_indices_from(c)] += 1e-5
        Lg = [np.linalg.cholesky(c) for c in Cg]

        invcov = block_diag(*[
            cho_solve((L, True), np.eye(self.dim.N*self.dim.R))
            for L in Lg])
        
        _A += invcov / lam
        _b = (S1.ravel() - v0.dot(np.kron(S0, np.eye(self.dim.N*self.dim.K)))).dot(V)

        #A += np.row_stack([inv_cov]*self.dim.K)
        _mean = np.linalg.lstsq(_A, _b, rcond=None)
            
        mean = np.linalg.lstsq(A, z1 - b, rcond=None)

        rank = mean[2]
        mean = mean[0]
        
        #mean = np.linalg.solve(ic, pm)

        fig, ax = plt.subplots()
        ax.plot(self.ttc, _mean[0], '-.')
        #ax.plot(self.ttc, mean, '--')
        #ax.plot(self.ttc, g, '+')
        ax.plot(self.ttc, gtrue, 'k-', alpha=0.4)
        plt.show()
        assert(False)
        return _mean[0]
        
    def foo(self, g, beta, mu_ivp,
            alpha_prec, responsibilites,
            ifix):

        N_repl = len(self.y_train_)

        resp = [np.row_stack([r_q]*self.dim.K)
                for r_q in responsibilites]

        inds = [np.concatenate([self.data_inds[m] + k*self.dim.N
                                for k in range(self.dim.K)])
                for m in range(N_repl)]

        ones = np.ones(self.dim.N)  # precompute a useful vector of ones

        def objfunc(g):
            ifx = 0

            Ki, Ki_grad = self._K(g, beta, ifx, eval_gradient=True)
            return Ki_grad
            #means = np.kron(mu_ivp[:, 0, :],
            #                ones).T
            #L, Lgrad = Mpower_action_wgrad(Ki, Ki_grad, means, self.order)
            #
            #return Lgrad

        def bar(g):
            x = np.kron(mu_ivp[:, 0, :],
                        ones).T
            # X[M_repl, K, N]  X[n, k, :]  kth dimension component of X
            X = x.T.reshape((N_repl, self.dim.K, self.dim.N))
            X = X.transpose((0, 2, 1))  # transpose the last two axes of X
            Arr = np.dstack([sum(brd*Ld
                                 for brd, Ld in zip(br, self.basis_mats)).T
                             for br in beta])

            # XA[n, ..., r] = X[n, ...].dot(A[...r])
            XA = np.einsum('ijk,klr->ijlr', X, Arr)


            W = self._get_weight_matrix(self.ttc, 0)
            # Want to store the outer product of the nth column of W
            # with the nth row X[, n, ]
            n = 3
            d = 1
            # broadcasting
            #     W        XA
            #   (N, N)   (M, N, K, R)
            #
            # WXA[m, n, :, k, r] = outer prod of the nth col. of W with
            # the mth replicate of X
            WXA = (W.T)[None, :, :, None, None] * XA[:, :, None, :, :]

            # finally we want to transpose and then reshape so that X is
            # vectorised
            WXA = WXA.transpose((0, 1, 4, 3, 2))
            return WXA.reshape((N_repl, self.dim.N, self.dim.R+1, self.dim.N*self.dim.K))

        def bar2(g, x):
            xshape = x.shape
            naxes = len(x.shape) - 1
            X = x.reshape(xshape[:-1] + (self.dim.K, self.dim.N))
            X = X.transpose(tuple(range(naxes)) + (naxes+1, naxes))

            Arr = np.stack([sum(brd*Ld
                                for brd, Ld in zip(br, self.basis_mats))
                            for br in beta])
            XA = np.einsum('...jk,...dlk->...djl', X, Arr)
            W = self._get_weight_matrix(self.ttc, 0)
            WXA = (W.T)[:, None, :, None] * XA[..., None, :, :, :]

            nshape = len(WXA.shape)
            perm = tuple(range(nshape-2)) + (nshape-1, nshape-2)
            WXA = WXA.transpose(perm)

            # Transpose and flatten the last two axes
            WXA = WXA.reshape(WXA.shape[:-2] + (self.dim.K*self.dim.N, ))
            return WXA

        def bar3(g, x):
            """
            Returns K[g, beta].dot(x[..., ]) and optionaly its gradient

            g_grad : shape (..., N, R, NK)
                g_grad[..., n, r, :] gives the gradient of K.dot(x[..., :])
                w.r.t g_{r, n}

            b_grad : shape (..., R, D, NK)
                b_grad[..., r, d, :] gives the gradient of K.dot(x[..., :])
                w.r.t. to beta_{r, d}
            """
            xshape = x.shape
            naxes = len(x.shape) - 1

            X = x.reshape(xshape[:-1] + (self.dim.K, self.dim.N))
            X = X.transpose(tuple(range(naxes)) + (naxes+1, naxes))

            XL = np.einsum('...jk,...dlk->...djl', X, self.basis_mats)
            W = self._get_weight_matrix(self.ttc, 0)
            WXL = (W.T)[:, None, :, None] * XL[..., None, :, :, :]

            nshape = len(WXL.shape)
            perm = tuple(range(nshape-2)) + (nshape-1, nshape-2)
            WXL = WXL.transpose(perm)

            # Transpose and flatten the last two axes
            WXL = WXL.reshape(WXL.shape[:-2] + (self.dim.K*self.dim.N, ))

            # multiply by beta along D and make a new axis for R


            # To get the gradient of the action with respect to
            # the latent forces multiply be beta and then sum over
            # d = 1,..,D
            WXA = beta[..., None] * WXL[..., None, :, :]            
            WXA = WXA.sum(3)

            # to get the gradient of the action wrt to
            # beta, multiply by G and then sum over n
            dB = G[..., None, None] * WXL[..., None, :, :]
            dB = dB.sum(1)
            return WXA, dB
            

        D1 = objfunc(g)
        x = np.kron(mu_ivp[:, 0, :], ones).T

        m = 1
        n = 4
        val1 = D1[n, ...].dot(x[:, m])

        D2 = bar(g)
        val2 = D2[m, n, 1, :]

        K = self._K(g, beta, 0)
        _Kx = K.dot(x)

        ifx = 0
        # rip out the fixed initial condition
        fixedval = x[ifx::self.dim.N, :].T
        xfix = np.kron(fixedval, ones).T

        G = np.column_stack((ones, g.reshape(self.dim.R, self.dim.N).T))
        Kx = (G[None, ..., None]) * D2
        Kx = Kx.sum((1, 2))
        Kx += xfix.T

        y = x.T
        # add a dummy extra axis
        
        
        D3 = bar2(g, y)
        D4, dB = bar3(g, y)

        def _foo(b):
            K = self._K(g, b, 0)
            return K.dot(y.T).T

        res = _foo(beta)
        r = 1
        d = 1
        eps = 1e-6
        betap = beta.copy()
        betap[r, d] += eps
        f = _foo(beta)
        fp = _foo(betap)
        dnum = (fp-f)/eps


        z = np.random.normal(size=y.size).reshape(y.shape)


        K, dK = self._K(g, beta, 0, True)


        power = 20
        import time
        t0 = time.time()
        pow2, _ = Mpower_action_wgrad(K, dK, y.T, 2)
        t1 = time.time()
        print("Old method:", t1 - t0)

        t0 = time.time()
        pow1, powergrad = self._Kx_power(g, beta, 0, y, power)
        t1 = time.time()
        print("New method:", t1 - t0)



class MLFMSA(EMFitMixin, BaseMLFMSA):

    def __init__(self, *args, **kwargs):
        super(MLFMSA, self).__init__(*args, **kwargs)


    def fit(self, times, Y, multi_output=False, **kwargs):

        self._setup_times(times,
                          kwargs.pop('h', None),
                          multi_output=multi_output)


        
        


    def _fit_init(self, is_fixed_vars, **kwargs):
        """
        Handles initalisation of the optimisation
        """
        init_strategies = {
            'g': lambda : np.zeros(self.dim.N*self.dim.R),
            'beta': lambda : np.row_stack((np.zeros(self.dim.D),
                                           np.eye(self.dim.R, self.dim.D))).ravel(),
            'v': lambda : -np.ones(sum(self.softmax_activs.param_shape))
            }

        var_names = ['g', 'beta', 'v']
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

