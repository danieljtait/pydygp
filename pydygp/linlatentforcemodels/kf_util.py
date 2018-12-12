import inspect
import numpy as np

def preprocess_arguments(argsets, converters):
    """convert and collect arguments in order of priority

    Parameters
    ----------
    argsets : [{argname: argval}]
        a list of argument sets, each with lower levels of priority
    converters : {argname: function}
        conversion functions for each argument

    Returns
    -------
    result : {argname: argval}
        processed arguments
    """
    result = {}
    for argset in argsets:
        for (argname, argval) in argset.items():
            # check that this argument is necessary
            if not argname in converters:
                raise ValueError("Unrecognized argument: {0}".format(argname))

            # potentially use this argument
            if argname not in result and argval is not None:
                # convert to right type
                argval = converters[argname](argval)

                # save
                result[argname] = argval

    # check that all arguments are covered
    if not len(converters.keys()) == len(result.keys()):
        missing = set(converters.keys()) - set(result.keys())
        s = "The following arguments are missing: {0}".format(list(missing))
        raise ValueError(s)

    return result

def array1d(X, dtype=None, order=None):
    """Returns at least 1-d array with data from X"""
    return np.asarray(np.atleast_1d(X), dtype=dtype, order=order)


def array2d(X, dtype=None, order=None):
    """Returns at least 2-d array with data from X"""
    return np.asarray(np.atleast_2d(X), dtype=dtype, order=order)


def get_params(obj):
    """Get names and values of all parameters in `objs' __init__ """
    try:
        # get names of every variable in the argument
        args = inspect.getargspec(obj.__init__)[0]
        args.pop(0)   # remove "self"

        # get values for each of the above in the object
        argdict = dict([(arg, obj.__getattribute__(arg)) for arg in args])
        return argdict

    except :
        raise ValueError("object has no __init__ method")


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('{0} cannot be used to seed a numpy.random.RandomState'
                     + ' instance').format(seed)
