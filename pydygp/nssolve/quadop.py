from . import ns_util
from . import linalg_util as la
import numpy as np

class QuadOperator:
    """
    I'm the base class!
    """
    def __init__(self, method=None):
        self.method = method


    def __new__(cls, method=None, **kwargs):
        if method == 'single_fp':
            """
            Allows the single point operator to be
            initalised from the general class
            """
            return QuadOperator_sfp(**kwargs)

class QuadOperator_sfp:

    @property
    def vecx_to_ravx(self):
        try:
            return self._vecx_to_ravx
        except:
            T = la.vecx_to_ravx(self.N, self.K)
            self._vecx_to_ravx = T
            return T

    """
    I'm the basic operator leaving one point fixed
    """
    
    method='single_fp'
    
    """
    A basic implementation of the operator with only a single
    fixed point
    """
    def __init__(self, intervals=None, fp_ind=0, K=1, R=1,
                 struct_mats=None, is_x_vec=False):

        self.is_x_vec = is_x_vec

        self.intervals = intervals
        self.fp_ind = fp_ind
        self.struct_mats = struct_mats

        # set the dimensions
        self.K = K
        self.R = R
        self.N = ns_util.get_times(self.intervals).size

        self.init_g_transform()

    def x_transform(self, gg, is_g_vec=True,
                    is_x_input_vec=False,
                    is_x_output_vec=None):
        if is_x_output_vec is None:
            is_x_output_vec = self.is_x_vec

        """
        Returns the operator :math:`K=K(g)` which transforms
        the values of x
        """
        if is_g_vec:
            N = self.N

            # glist equals the `columns` of
            # G = [ g_1(t) | g_2(t) | ... g_r(t) ]
            glist = [g for g in gg.reshape(self.R, N)]

            # Operator defined to act on x.ravel()
            res = ns_util.quad_xop_intervals(self.fp_ind,
                                             self.intervals,
                                             self.struct_mats,
                                             glist,
                                             self.K)

            # Transform the operator to act on vec(x) instead
            if is_x_output_vec:
                # get the transformation vecx --> x.ravel()
                T = la.vecx_to_ravx(N, self.K)
                res = np.dot(T.T, res)

                if is_x_input_vec:
                    res = np.dot(res, T)

            return res

    def g_transform(self, xx, is_x_input_vec=False, is_x_output_vec=False):
        if is_x_input_vec:
            # xx = X.T.ravel()
            #  -> we need to reshape it to xx = X.ravel()
            xx = xx.reshape(self.K, self.N).T.ravel()

            XtxI = kron_A_N(xx[None, :], self.N*self.K)
#        XtxI = np.kron(xx, np.identity(self.N*self.K))


        L = np.dot(XtxI, self._K_vecg[0])
        b = np.dot(XtxI, self._K_vecg[1])

        if is_x_output_vec:
            # output is in X.ravel() form, needs to be in vectorised
            T = self.vecx_to_ravx
            L = np.dot(T.T, L)
            b = np.dot(T.T, b)

        return L, b        

    def init_g_transform(self, is_x_vec=False):

        # calculate the derivative for fixed form of vec(K)
        is_x_output_vec = False
        is_x_input_vec = False

        g_zero = np.zeros((self.R, self.N))

        # first construct the pair dKdg dc0
        const = self.x_transform(g_zero.ravel(),
                                 is_x_input_vec=is_x_input_vec,
                                 is_x_output_vec=is_x_output_vec).T.ravel()  # vec(K(0))

        cols = []
        for r in range(self.R):
            for n in range(self.N):
                g_zero[r, n] = 1.
                cols.append(self.x_transform(g_zero.ravel(),
                                             is_x_input_vec=is_x_input_vec,
                                             is_x_output_vec=is_x_output_vec).T.ravel() - const)
                g_zero[r, n] = 0.

        dKdg = np.column_stack(cols)

        self._K_vecg = (dKdg, const)

        
        
def kron_A_N(A, N):  # Simulates np.kron(A, np.eye(N))
    m,n = A.shape
    out = np.zeros((m,N,n,N),dtype=A.dtype)
    r = np.arange(N)
    out[:,r,:,r] = A
    out.shape = (m*N,n*N)
    return out