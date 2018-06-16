import numpy as np
from pydygp import nssolve
import scipy.sparse as sparse
from collections import namedtuple
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal

# Model dimensions:
#   N - size of augmented time vector
#   K - dimension of the ambient space
#   R - number of latent forces
Dimensions = namedtuple('Dimensions', 'N K R')

class MLFM_NS_EM:
    """
    Carries out EM estimation of the latent forces in the MLFM model.
    """
    def __init__(self,struct_mats, order=1):
        self.order = order
        self.struct_mats = np.asarray(struct_mats)

        # model dimensions
        self.dim = Dimensions(N=None,
                              K=self.struct_mats[0].shape[0],
                              R=len(self.struct_mats)-1)        


    """
    Model Setup functions
    *********************
    """
    
    def time_input_setup(self, data_times, h=None):
        """
        Sets up the intervals and handles the number of times in each interval
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
        """
        Setup for the successive approximation operator.
        """
        self.t0 = self.intervals[ifix].ta
        nsop = nssolve.QuadOperator(fp_ind=ifix,
                                    method='single_fp',
                                    intervals=self.intervals,
                                    K=self.dim.K,
                                    R=self.dim.R,
                                    struct_mats=self.struct_mats,
                                    is_x_vec=True)
        self.nsop = nsop        

    """
    Initalise model parameters
    """
    def init_beta(self, beta=1e4):
        self.beta = beta

    def _em_init_meancov(self):
        scale = 0.1
        Y = self.vecy.reshape((self.dim.K, self.data_times.size))
        xinterp = []
        for yk in Y:
            xinterp.append(np.interp(self.comp_times, self.data_times, yk))
        xinterp = np.concatenate(xinterp)

        if self.order > 1:
            m = np.concatenate(*[xinterp]*self.order)
        else:
            m = xinterp

        c = np.diag(np.ones(m.size)*scale)
        return m, c

    def fit(self, liktol=1e-3, max_nt=100):

        # initalise the mean and covariance of the latent var.
        m, c = self._em_init_meancov()

        gcur = self.Mstep(m, c)
        llcur = self.loglikelihood(self.vecy, gcur)

        nt = 0
        ldelta = 1

        while ldelta > liktol:
            m, c = self.Estep(gcur)
            gnew = self.Mstep(m, c)

            llnew = self.loglikelihood(self.vecy, gnew)

            # check change in ll and update everything
            ldelta = llnew - llcur
            gcur = gnew
            llcur = llnew

            nt += 1
            if nt >= max_nt:
                print("Maximum number of function evaluations exceeded")
                break

        return gcur
            
        
    def Estep(self, g):
        """
        Returns the mean and covariance of ln p(z | y, g)
        """
        b = self.beta
        D = self.sparse_data_map

        # sparse representation of the transformation matrix
        K = sparse.coo_matrix(self.nsop.x_transform(g, is_x_input_vec=True))

        KtK = K.T.dot(K)
        DK = D.dot(K)
        Ink = sparse.eye(self.dim.N*self.dim.K)

        # for padding the block diag
        zeros = sparse.coo_matrix(np.zeros((self.dim.N*self.dim.K,
                                            self.dim.N*self.dim.K)))

        L0 = self.Lambda0
        LM = self.LambdaM

        if self.order == 1:
            inv_covar = L0 + DK.T.dot(LM.dot(DK))
            
            pre_mean = DK.T.dot(LM.dot(self.vecy[:, None]))
            cov = sparse.linalg.spsolve(inv_covar, sparse.eye(pre_mean.size))
            #mean = cov.dot(pre_mean)

            mean = sparse.linalg.spsolve(inv_covar, pre_mean)
            return mean, cov.toarray()

    def Mstep(self, z_mean, z_cov):
        """
        Maximises E[ln p(y, z | g)] + ln p(g) for the force g.
        """
        # retrive the data map and x0 inv cov
        D = self.sparse_data_map
        
        # calc. Ezzt
        Ezzt = z_cov + np.outer(z_mean, z_mean)

        # get y conditional distribution cov
        y = self.vecy
        LM = self.LambdaM
        LMy = LM.dot(y)
        vecLMy = sparse.coo_matrix(LMy.T.ravel())

        DtL1D = D.T.dot(LM.dot(D))

        # Kron product Ezzt x DtL1D
        Ezzt_x_DtL1D = sparse.kron(Ezzt, DtL1D)
        mt_x_D = sparse.kron(z_mean[None, :], D)

        P, p = self.vecK_aff_rep

        g_inv_cov = P.T.dot(Ezzt_x_DtL1D)
        g_inv_cov = g_inv_cov.dot(P).toarray() # no longer sparse at this point
        g_inv_cov += self.gprior_inv_cov

        g_pre_mean = vecLMy.dot(mt_x_D)
        g_pre_mean -= p.dot(Ezzt_x_DtL1D)
        g_pre_mean = P.T.dot(g_pre_mean.T).toarray() # no longer sparse at this point

        g_mean = np.linalg.solve(g_inv_cov, g_pre_mean)
        return g_mean.ravel()
        

    def loglikelihood(self, y, g):
        """
        Returns the model log-likelihood
        """
        K = self.nsop.x_transform(g, is_x_input_vec=True)

        # recursively construct the cov. function
        cov = self.xcov

        binvI = np.eye(cov.shape[0])/self.beta

        for M in range(self.order):
            cov = np.dot(K, np.dot(cov, K.T)) + binvI

        # data transform
        D = self.data_map
        cov = np.dot(D, np.dot(cov, D.T)) + self.data_covar

        lpy_g = multivariate_normal.logpdf(y, np.zeros(y.size), cov)
        lpg = multivariate_normal.logpdf(g, np.zeros(g.size), self.gcov)

        return lpy_g + lpg

    @property
    def data_map(self):
        """
        Matrix that maps the augmented data x_N to the data
        for which there is an observed time point
        """
        data_map = np.row_stack((np.eye(N=1, M=self.dim.N, k=i)
                                 for i in self.data_inds))
        data_map = block_diag(*[data_map]*self.dim.K)
        return data_map


    @property
    def sparse_data_map(self):
        """
        Sparse representation of the linear transformation y = Dx_N
        """
        try:
            return self._s_data_map
        except:
            D = self.data_map
            self._s_data_map = sparse.coo_matrix(D)
            return self._s_data_map

    @property
    def Lambda0(self):
        try:
            return self._L0
        except:
            xinvs = [sparse.coo_matrix(ci) for ci in self.xcov_invs]
            self._L0 = sparse.block_diag(*[xinvs])
            return self._L0

    @property
    def LambdaM(self):
        try:
            return self._LM
        except:
            Cy = self.data_covar
            Cm = sparse.eye(Cy.shape[0])/self.beta
            LM = sparse.linalg.spsolve(Cy + Cm, sparse.eye(Cy.shape[0]))
            self._LM = LM
            return LM


    @property
    def vecK_aff_rep(self):
        """
        Pair A, b such that vec(K[g]) = Ag + b
        """
        try:
            return self._vecK_aff_rep
        except:
            NR = self.dim.N*self.dim.R
            
            const = self.nsop.x_transform(np.zeros(NR),
                                          is_x_input_vec=True).T.ravel()
            const = sparse.coo_matrix(const)


            A = sparse.hstack([sparse.coo_matrix(
                self.nsop.x_transform(np.eye(N=1,
                                             M=NR,
                                             k=i).ravel(),
                                      is_x_input_vec=True).T.ravel()[:, None] - const.T)
                               for i in range(NR)])

            self._vecK_aff_rep = A, const
            return A, const
