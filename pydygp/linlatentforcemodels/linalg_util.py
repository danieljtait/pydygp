import numpy as np


# decorator to take a function computing
# the cholesky decomposition and add a small
# noise term to avoid...
def jitter(cholfunc):

    def wrapper(*args, **kwargs):
        eps = 1e-4
        max_n = 10
        
        try:
            eps = kwargs['eps']
        except:
            pass
        
        try:
            result = cholfunc(*args, **kwargs)
            return result

        except:
            x = args[0]
            rargs = args[1:]

            n = 0
            while n < max_n:
                x += eps*np.diag(np.ones(x.shape[0]))
                try:
                    L = cholfunc(x, *rargs, **kwargs)
                    return L
                except:
                    n += 1 # incr. the attempt count

            raise ValueError("Failed to compute the cholesky decomposition")
    return wrapper


@jitter
def cholesky(x, **kwargs):
    return np.linalg.cholesky(x)

def back_sub(L, x):
    """
    Returns the result :math:`\mathbf{C}^{-1}\mathbf{x}` using
    the back substition
    .. math:
       x = 
    """
    return np.linalg.solve(L.T, np.linalg.solve(L, x))
