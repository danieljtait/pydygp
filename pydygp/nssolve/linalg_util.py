import numpy as np
import scipy.linalg

def block_diag(*args):
    return scipy.linalg.block_diag(*args)

def block_diag_intersect_f(A, B, dim):
    """
    For matrices :math:`\mathbf{A}, \mathbf{B}` this function returns
    the block diagonal matrix of A, B where the lower-right 'dim' submatrix
    of :math:`\mathbf{A}` overwrites the corresponding upper-left submatrix
    of :math:`\mathbf{B}`.
    """
    B11 = B[:dim[0], :dim[1]]
    B12 = B[:dim[0], dim[1]:]
    B21 = B[dim[0]:, :dim[1]]
    B22 = B[dim[0]:, dim[1]:]

    res = scipy.linalg.block_diag(A, B22)
    res[A.shape[0]-dim[0]:A.shape[0], A.shape[1]: ] = B12
    res[A.shape[0]:, A.shape[1]-dim[1]:A.shape[1]] = B21
    return res


def block_diag_intersect_b(A, B, dim):
    """
    For matrices :math:`\mathbf{A}, \mathbf{B}` this function returns
    the block diagonal matrix of A, B where the upper-left 'dim' submatrix
    of :math:`\mathbf{B}` overwrites the corresponding lower-right submatrix
    of :math:`\mathbf{A}`.
    """
    A11 = A[:-dim[0], :-dim[1]]
    A12 = A[:-dim[0], -dim[1]:]
    A21 = A[-dim[0]:, :-dim[1]]
    A22 = A[-dim[0]:, -dim[1]:]

    res = scipy.linalg.block_diag(A11, B)
    res[:A.shape[0]-dim[0], A.shape[1]-dim[1]:A.shape[1]] = A12
    res[A.shape[0]-dim[0]:A.shape[0], :A.shape[1]-dim[1]] = A21
    return res


def vecx_to_ravx(N, K):
    """
    Returns the (NK)x(NK) matrix such that

    .. code::

       >>> x = np.arange(0, 5).reshape(3, 2)
       >>> y = np.dot(T, vecx)
       >>> assert(np.all(y == x.ravel())
    """
    res = np.row_stack([
        np.row_stack([np.eye(N=1, M=N*K, k=N*k + n) for k in range(K)])
        for n in range(N)])
    return res


def commutation_matrix(r, m):
    return sum(
        np.kron(np.outer(np.eye(N=1, M=r, k=i), np.eye(N=1, M=m, k=j)),
                np.outer(np.eye(N=1, M=m, k=j), np.eye(N=1, M=r, k=i)))
        for i in range(r) for j in range(m))    
