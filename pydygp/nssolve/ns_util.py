import numpy as np
from . import linalg_util as lau

class Interval:
    """
    Class to handle the performance of quadrature
    over the interval :math:`I = [t_a, t_b]`
    """
    def __init__(self, ta, tb):
        self.ta = ta
        self.tb = tb

    def set_quad_style(self, h=None, quad_type='trap_simp'):
        """
        handles the descretisation of the :class:`Interval` to
        allow for an approximation of the integral solution operator
        """
        if quad_type == 'trap_simp':
            if h is None:
                n = 3
            else:
                width = self.tb - self.ta
                n = np.ceil(width/h) + 1

                # simpsons rule requires at least 3
                # quadrature nodes
                n = max(n, 3)

            self.quad_type = 'trap_simp'
            self.tt = np.linspace(self.ta, self.tb, n)

        else:
            raise NotImplementedError('only Trapezoidal/Simpson quadrature currently supported')


def get_times(intervals):
    """
    Utility function to concatenate the time vectors for an ordered
    list of :class:`Interval` objects respecting the fact the shared
    end points
    """
    tt = intervals[0].tt
    for I in intervals[1:]:
        tt = np.concatenate((tt, I.tt[1:]))
    return tt

def quad_xop_intervals(i, intervals, model_mats, glist, dim, return_grad=False):
    """
    Returns the `glued together' representation of the operator
    :class:`f_quad_xop_interval` for the a collection of
    intervals :math:`I_1,\ldots,I_n` with
    :math:`I_{i} = [t_{a_i}, t_{b_i}]` with matching end points
    :math:`t_{b_{i}} = t_{a_{i+1}}` 
    """
    if i == -1:
        i = len(intervals)

    nt = 0
    _G = np.column_stack(glist)

    res = None

    for count, I in enumerate(intervals):

        ni = I.tt.size
        Gi = _G[nt:nt+ni, :]
        
        if count < i:
            """
            get the operator solved backward over the interval I
            """
            op_I = b_quad_xop_interval(I, model_mats, Gi, dim)

            # combine with existing
            if res is None:
                res = op_I
            else:
                res = lau.block_diag_intersect_b(res, op_I, (dim, dim))

        else:
            op_I = f_quad_xop_interval(I, model_mats, Gi, dim)

            # combine with existing
            if res is None:
                res = op_I
            else:
                res = lau.block_diag_intersect_f(res, op_I, (dim, dim))

        nt += ni - 1

    if return_grad:
        # returns the pair A,b such that
        # if Ag + b = vec(K)
        R = len(model_mats)-1
        N = glist[0].size  # implicitly assumes g is not None
        
        b = quad_xop_intervals(i,
                               intervals, model_mats,
                               [np.zeros(N)]*R,
                               dim).T.ravel()
        
        cols = []
        zero_list = [np.zeros(N)]*R
        for r in range(R):
            for n in range(N):
                zero_list[r][n] = 1.
                col = quad_xop_intervals(i,
                                         intervals, model_mats,
                                         zero_list,
                                         dim).T.ravel() - b
                zero_list[r][n] = 0.
                cols.append(col)
        A = np.column_stack(cols)
        return res, A, b
        
    else:
        return res

def f_quad_xop_interval(interval, model_mats, G, dim):
    """
    returns the operator approximating the integral

    .. math::
       x(t_i) = \int_{t_a}^{t_b} A(\\tau)x(\\tau)\operatorname{d}\\tau

    over the interval :math:`I=[t_a, t_b]`.
    """

    # number of time points
    Nt = interval.tt.size

    # get the value of the flow function
    As = [model_mats[0] + sum(gr*Ar for gr, Ar in zip(g, model_mats[1:]))
                              for g in G]

    # initalise a result matrix
    res = np.zeros((dim*Nt, dim*Nt))

    # `dim` sized identity matrix
    Id = np.eye(dim)

    if interval.quad_type == 'trap_simp':

        # First row is simply the identity transform
        res[:dim, :dim] = Id

        # Second row is given by the trapezoidal approximation
        wt = (interval.tt[1] - interval.tt[0])/2  # trap. quad weight

        res[dim:2*dim, :dim] = Id + wt*As[0]
        res[dim:2*dim, dim:2*dim] = wt*As[1]

        # Remaining rows are given by Simpson's rule
        for rn in range(2, Nt):
            ws = (interval.tt[rn]-interval.tt[rn-2])/6.  # simp. quad weight

            res[rn*dim:(rn+1)*dim, (rn-2)*dim:(rn-1)*dim] = Id + ws*As[rn-2]
            res[rn*dim:(rn+1)*dim, (rn-1)*dim:(rn)*dim] = 4*ws*As[rn-1]
            res[rn*dim:(rn+1)*dim, rn*dim:(rn+1)*dim] = ws*As[rn]

        return res

    else:
        raise NotImplementedError('quad type must be set to trap_simp')


def b_quad_xop_interval(interval, model_mats, G, dim):
    """
    returns the operator approximating the integral

    .. math::
       x(t_i) = x(t_b) + \int_{t_b}^{t_a}

    over the interval :math:`I=[t_a, t_b]` such that
    """

    # number of time points
    Nt = interval.tt.size

    # get the value of the flow function
    As = [model_mats[0] + sum(gr*Ar for gr, Ar in zip(g, model_mats[1:]))
          for g in G]

    # initalise a result matrix
    res = np.zeros((dim*Nt, dim*Nt))

    # `dim` sized identity matrix
    Id = np.identity(dim)

    if interval.quad_type == 'trap_simp':

        # Final row is the identity transform
        res[-dim:, -dim:] = Id

        # Trapezoidal for penultimate row
        wt = (interval.tt[-2] - interval.tt[-1])/2 # trap. quad weight

        res[-2*dim:-dim, -dim:] = Id + wt*As[-1]
        res[-2*dim:-dim, -2*dim:-dim] = wt*As[-2]

        # Remaining rows are given by Simplson's rule
        for rn in range(2, Nt):
            ws = (interval.tt[Nt-rn-1] - interval.tt[Nt-(rn-1)])/6

            ia = Nt-rn-1
            ib = Nt-rn+1
            res[ia*dim:(ia+1)*dim, ia*dim:(ia+1)*dim] = ws*As[ia]
            res[ia*dim:(ia+1)*dim, (ia+1)*dim:(ia+2)*dim] = 4*ws*As[ia+1]
            res[ia*dim:(ia+1)*dim, ib*dim:(ib+1)*dim] =  Id + ws*As[ib]

        return res
    
    else:
        raise NotImplementedError('quad type must be set to trap_simp')



def f_quad_gop_interval(interval, X, dim):

    if interval.quad_type == 'trap_simp':

        wt = (interval.tt[1] - interval.tt[0])/2
        Nt = interval.tt.size

        P = np.zeros((dim*Nt, Nt))

        P[:dim, 1] = wt*X[0, :]
        P[dim:2*dim, 1] = wt*X[1, :]

        for rn in range(2, Nt):
            ws = (interval.tt[rn]-interval.tt[rn-2])/6
            col = np.concatenate((ws*X[rn-2, :],
                                  4*ws*X[rn-1, :],
                                  ws*X[rn, :]))
            P[(rn-2)*dim:(rn+1)*dim, rn] = col

        b = np.row_stack([X[0, ] for i in range(Nt)])

        return P.T, b

def b_quad_gop_interval(interval, X, dim):

    if interval.quad_type == 'trap_simp':

        Nt = interval.tt.size

        P = np.zeros((dim*Nt, Nt))

        wt = (interval.tt[-2] - interval.tt[-1])/2
        P[-2*dim:-dim, -2] = wt*X[-2, :]
        P[-dim: -2] = wt*X[-1, :]

        for rn in range(2, Nt):

            ws = (interval.tt[Nt-rn-1] - interval.tt[Nt-(rn-1)])/6
            ia = (Nt - rn - 1)*dim
            ib = ia + 3*dim

            col_ind = -(rn+1)
            col = np.concatenate((ws*X[-(rn+1), :],
                                  4*ws*X[-(rn), :],
                                  ws*X[-(rn-1), :]))

            P[ia:ib, col_ind] = col

        b = np.row_stack([X[-1, ] for i in range(Nt)])
        return P.T, b


"""
Representaiton of the operator acting on g
"""

def vec_flow(model_mats, N):
    """
    Returns U, v such that Uvec(g) + v = vec([A(t_0) | ... | A(t_N-1)])
    """
    A = np.column_stack((
        lau.block_diag(*[Ar.T.ravel()[:, None]]*N) for Ar in model_mats[1:]))
    b = np.concatenate([model_mats[0].T.ravel()]*N)
    return A, b


def quad_gop_intervals(i, intervals, model_mats, xlist, dim):
    """
    Returns a linear operator A and a vector b such that

    Ag + b approx vec( x0 + int_0^t A(tau)x(tau) dtau )

    note this vectorises X, and acts on vectorised G, unlike the representation
    in xop
    """

    if i == -1:
        i = len(intervals)

    nt = 0
    _X = np.column_stack(xlist)
    N = _X.shape[0]
    
    resP = None
    resb = None

    for count, I in enumerate(intervals):

        ni = I.tt.size
        Xi = _X[nt:nt+ni, :]

        if count < i:
            """
            Get the operator acting on the flow matrices
            backward over i
            """
            Fop_I, Fb_I = b_quad_gop_interval(I, Xi, dim)

            if resP is None:
                resP = Fop_I
                resb = Fb_I
            else:
                resP = lau.block_diag_intersect_b(resP, Fop_I, (1, dim))
                resb = np.row_stack((resb[:-1 ], Fb_I))

        else:
            Fop_I, Fb_I = f_quad_gop_interval(I, Xi, dim)
            if resP is None:
                resP = Fop_I
                resb = Fb_I
            else:
                resP = lau.block_diag_intersect_f(resP, Fop_I, (1, dim))
                resb = np.row_stack((resb, Fb_I[1:, ]))

        nt += ni - 1

    # Get the representation of vec([A(0) | ... | A(N-1)])^T in terms of
    # g
    
    Aflow, bflow = vec_flow(model_mats, N)

    # We want vec(flowT) so we get the commutation matrix
    Krm = lau.commutation_matrix(dim, dim*N)
    # And now 
    Aflow = np.dot(Krm, Aflow)
    bflow = np.dot(Krm, bflow)

    IxP = np.kron(np.identity(dim), resP)
    resP = np.dot(IxP, Aflow)
    resb = np.dot(IxP, bflow) + resb.T.ravel()
    
    return resP, resb
                
