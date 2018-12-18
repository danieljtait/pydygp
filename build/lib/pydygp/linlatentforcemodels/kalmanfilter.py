"""
Credit for almost all of this code goes to the authors of the
PyKalman package
"""
import numpy as np
from . kf_util import array1d, array2d, preprocess_arguments, get_params, check_random_state


def _handle_shape(X, t, name):
    """My version of the shape extracting function. """
    if name == 'obs_mat':
        if len(X.shape) == 4:
            return X[t, ...]
        else:
            return X

def _last_dims(X, t, ndims=2):
    """Extract the final dimensions of `X`

    Extract the final `ndim` dimensions at index `t` if `X` has >= `ndim` + 1
    dimensions, otherwise return `X`.

    Parameters
    ----------
    X : array with at least dimension `ndims`
    t : int
        index to use for the `ndims` + 1th dimension
    ndims : int, optional
        number of dimensions in the array desired

    Returns
    -------
    Y : array with dimension `ndims`
        the final `ndims` dimensions indexed by `t`
    """
    X = np.asarray(X)
    if len(X.shape) == ndims + 1:
        return X[t]
    elif len(X.shape) == ndims:
        return X
    else:
        raise ValueError(("X only has %d dimensions when %d" +
                " or more are required") % (len(X.shape), ndims))

def _filter_predict(transition_matrix, transition_covariance,
                    transition_offset, current_state_mean,
                    current_state_covariance):
    """

    """
    predicted_state_mean = np.dot(transition_matrix, current_state_mean) + \
                           transition_offset

    #predicted_state_covariance = np.dot(transition_matrix,
    #                                    np.dot(current_state_covariance,
    #                                           transition_matrix.T))

    predicted_state_covariance = np.einsum('ij,mjk->mik', transition_matrix,
                                           np.einsum('mij,kj->mik',
                                                     current_state_covariance,
                                                     transition_matrix))
    
    predicted_state_covariance += transition_covariance[None, ...]

    return predicted_state_mean, predicted_state_covariance

def _filter_correct(observation_matrix, observation_covariance,
                    observation_offset, predicted_state_mean,
                    predicted_state_covariance, observation):

    if not np.any(np.ma.getmask(observation)):
        #predicted_observation_mean = np.dot(observation_matrix,
        #                                    predicted_state_mean) + \
        #                                    observation_offset
        predicted_observation_mean = np.einsum('mij,jm->im',
                                               observation_matrix,
                                               predicted_state_mean) #+ \
                                               #observation_offset
                                
        
        #predicted_observation_covariance = np.dot(observation_matrix,
        #                                          np.dot(predicted_state_covariance,
        #                                                 observation_matrix.T)) + \
        #                                                 observation_covariance
        predicted_observation_covariance = np.einsum('mij,mjk->mik',
                                                     observation_matrix,
                                                     np.einsum('mij,mkj->mik',
                                                               predicted_state_covariance,
                                                               observation_matrix))
        predicted_observation_covariance += observation_covariance
        
        #kalman_gain = np.dot(predicted_state_covariance,
        #                     np.dot(observation_matrix.T,
        #                            np.linalg.pinv(predicted_observation_covariance)))

        # pinv seems to handle stacked matrices naturally
        pinv = np.linalg.pinv(predicted_observation_covariance)
        kalman_gain = np.einsum('mij,mjk->mik',
                                predicted_state_covariance,
                                np.einsum('mji,mjk->mik',
                                          observation_matrix,
                                          pinv))
        # This will handle the case where predicted_observation_mean is
        # shape [n_dim_state, ...]
        #corrected_state_mean = predicted_state_mean + \
        #                       kalman_gain.dot(observation - predicted_observation_mean)
        corrected_state_mean = predicted_state_mean + \
                               np.einsum('mij,jm->im', kalman_gain,
                                         observation - predicted_observation_mean)
        #corrected_state_covariance = predicted_state_covariance - \
        #                             np.dot(kalman_gain,
        #                                    np.dot(observation_matrix,
        #                                           predicted_state_covariance))
        corrected_state_covariance = predicted_state_covariance - \
                                     np.einsum('mij,mjk->mik', kalman_gain,
                                               np.einsum('mij,mjk->mik',
                                                         observation_matrix,
                                                         predicted_state_covariance))
    else:
        # no observation to worry about
        n_dim_state = predicted_state_covariance.shape[-1]
        n_dim_obs = observation_matrix.shape[1]
        kalman_gain = np.zeros((n_dim_state, n_dim_obs))
        corrected_state_mean = predicted_state_mean
        corrected_state_covariance = predicted_state_covariance

    return kalman_gain, corrected_state_mean, corrected_state_covariance


def _filter(transition_matrices, observation_matrices, transition_covariance,
            observation_covariance, transition_offsets, observation_offsets,
            initial_state_mean, initial_state_covariance, observations):
    """Apply the Kalman Filter

    """
    n_timesteps = observations.shape[0]
    n_dim_state = len(initial_state_mean)
    n_dim_obs = observations.shape[1]

    n_samp = initial_state_mean.shape[-1]
    # use the initial state mean to learn the shape of the
    # state variable
    state_shape = initial_state_mean.shape 

    predicted_state_means = np.zeros((n_timesteps, ) + state_shape)
    predicted_state_covariances = np.zeros(
        (n_timesteps, state_shape[1], n_dim_state, n_dim_state))

    kalman_gains = np.zeros(
        (n_timesteps, state_shape[1], n_dim_state, n_dim_obs))

    filtered_state_means = np.zeros((n_timesteps, ) + state_shape)
    filtered_state_covariances = np.zeros(
        (n_timesteps, state_shape[1], n_dim_state, n_dim_state))

    for t in range(n_timesteps):
        if t == 0:
            predicted_state_means[t] = initial_state_mean
            predicted_state_covariances[t] = initial_state_covariance

        else:
            transition_matrix = _last_dims(transition_matrices, t-1)
            transition_covariance = _last_dims(transition_covariance, t - 1)
            #transition_offset = _last_dims(transition_offsets, t - 1, ndims=n_samp)
            transition_offset = 0.
            predicted_state_means[t], predicted_state_covariances[t] = (
                _filter_predict(
                transition_matrix,
                transition_covariance,
                transition_offset,
                filtered_state_means[t - 1],
                filtered_state_covariances[t - 1],
                )
                )
        observation_matrix = _handle_shape(observation_matrices, t, 'obs_mat')
        #observation_matrix = _last_dims(observation_matrices, t)
        #observation_matrix = observation_matrices

        
        observation_covariance = _last_dims(observation_covariance, t)
        observation_offset = _last_dims(observation_offsets, t, ndims=n_samp)
        observation_offset = 0.

        (kalman_gains[t], filtered_state_means[t],
         filtered_state_covariances[t]) = (
            _filter_correct(observation_matrix,
                            observation_covariance,
                            observation_offset,
                            predicted_state_means[t],
                            predicted_state_covariances[t],
                            observations[t]
                            )
            )

    return (predicted_state_means, predicted_state_covariances,
            kalman_gains, filtered_state_means,
            filtered_state_covariances)
                
def _smooth(transition_matrices, filtered_state_means,
            filtered_state_covariances, predicted_state_means,
            predicted_state_covariances):
    """ Apply the Kalman Smoother
    """
    n_timesteps, n_dim_state, n_outputs = filtered_state_means.shape

    smoothed_state_means = np.zeros((n_timesteps, n_dim_state, n_outputs))
    smoothed_state_covariances = np.zeros(
        (n_timesteps, n_outputs, n_dim_state, n_dim_state))

    kalman_smoothing_gains = np.zeros(
        (n_timesteps - 1, n_outputs, n_dim_state, n_dim_state))

    smoothed_state_means[-1] = filtered_state_means[-1]
    smoothed_state_covariances[-1] = filtered_state_covariances[-1]
    
    for t in reversed(range(n_timesteps - 1)):
        # gets the transition matrix at time t if the transition
        # matrix is inhomogenous, else returns the constant
        # transition matrix
        transition_matrix = _last_dims(transition_matrices, t)

        (smoothed_state_means[t], smoothed_state_covariances[t],
         kalman_smoothing_gains[t]) = \
         (
            _smooth_update(
            transition_matrix,
            filtered_state_means[t],
            filtered_state_covariances[t],
            predicted_state_means[t+1],
            predicted_state_covariances[t+1],
            smoothed_state_means[t+1],
            smoothed_state_covariances[t+1]
            )
        )

    return (smoothed_state_means,
            smoothed_state_covariances,
            kalman_smoothing_gains)

def _smooth_update(transition_matrix, filtered_state_mean,
                   filtered_state_covariance, predicted_state_mean,
                   predicted_state_covariance, next_smoothed_state_mean,
                   next_smoothed_state_covariance):
    """Correct a predicted state with a Kalman Smoother update."""
    pinv = np.linalg.inv(predicted_state_covariance)
    kalman_smoothing_gain = np.einsum('mij,mjk->mik',
                                      filtered_state_covariance,
                                      np.einsum('ij,mjk->mik',
                                                transition_matrix.T,
                                                pinv))

    smoothed_state_mean = filtered_state_mean + \
                          np.einsum('mij,jm->im', kalman_smoothing_gain,
                                    next_smoothed_state_mean - predicted_state_mean)

    smoothed_state_covariance = filtered_state_covariance + \
                                np.einsum('mij,mjk->mik',
                                          kalman_smoothing_gain,
                                          np.einsum('mij,mkj->mik',
                                                    next_smoothed_state_covariance - \
                                                    predicted_state_covariance,
                                                    kalman_smoothing_gain))

    return smoothed_state_mean, smoothed_state_covariance, kalman_smoothing_gain



class KalmanFilter(object):

    def __init__(self, transition_matrices=None, observation_matrices=None,
                 transition_covariance=None, observation_covariance=None,
                 transition_offsets=None, observation_offsets=None,
                 initial_state_mean=None, initial_state_covariance=None,
                 random_state=None,
                 n_dim_state=None, n_dim_obs=None):

        self.transition_matrices = transition_matrices
        self.observation_matrices = observation_matrices
        self.transition_covariance = transition_covariance
        self.observation_covariance = observation_covariance
        self.transition_offsets = transition_offsets
        self.observation_offsets = observation_offsets
        self.initial_state_mean = initial_state_mean
        self.initial_state_covariance = initial_state_covariance
        self.random_state = random_state
        self.n_dim_state = n_dim_state
        self.n_dim_obs = n_dim_obs


    def _smooth_pair(self, smoothed_state_covariances, kalman_smoothing_gains):
        """Calculate pairwise covariance between latent states
        """
        n_timesteps, n_outputs, n_dim_state, _ = smoothed_state_covariances.shape
        pairwise_covariances = np.zeros(
            (n_timesteps, n_outputs, n_dim_state, n_dim_state))
        for t in range(1, n_timesteps):
            pairwise_covariances[t] = \
                np.einsum('mij,mkj->mik',
                          smoothed_state_covariances[t],
                          kalman_smoothing_gains[t - 1])
        return pairwise_covariances

    def filter(self, X):
        """ Apply the Kalman Filter
        """
        Z = self._parse_observations(X)

        (transition_matrices, transition_offsets, transition_covariance,
         observation_matrices, observation_offsets, observation_covariance,
         initial_state_mean, initial_state_covariance) = (
            self._initialize_parameters()
        )

        (_, _, _, filtered_state_means,
         filtered_state_covariances) = (
            _filter(
                transition_matrices, observation_matrices,
                transition_covariance, observation_covariance,
                transition_offsets, observation_offsets,
                initial_state_mean, initial_state_covariance,
                Z
            )
        )
        return (filtered_state_means, filtered_state_covariances)
        
    def smooth(self, X):
        """Apply the Kalman Smoother
        Apply the Kalman Smooth to estimate the hidden state at time

        Parameters
        ----------
        X : [n_timesteps, n_dim_obs, n_outputs] array-like
            observations correspong to times [0,...,n_timesteps-1]. If
            `X` is a masked array and any of `X[t]` is masked, then `X[t]`
            will be treated as a missing observation.
        """
        Z = self._parse_observations(X)

        # Initialize the parameters
        (transition_matrices, transition_offsets, transition_covariance,
         observation_matrices, observation_offsets, observation_covariance,
         initial_state_mean, initial_state_covariance) = (
            self._initialize_parameters()
        )

        # Carry out the forward filtering
        (predicted_state_means, predicted_state_covariances,
         _, filtered_state_means, filtered_state_covariances) = (
            _filter(
                transition_matrices, observation_matrices,
                transition_covariance, observation_covariance,
                transition_offsets, observation_offsets,
                initial_state_mean, initial_state_covariance, Z
            )
        )

        # Backwards smoothing
        (smoothed_state_means, smoothed_state_covariances, smoothed_kalman_gains) = (
            _smooth(
                transition_matrices, filtered_state_means,
                filtered_state_covariances, predicted_state_means,
                predicted_state_covariances
            )[:3]
        )
        return (smoothed_state_means, smoothed_state_covariances, smoothed_kalman_gains)


    def _initialize_parameters(self):
        """Retrieve parameters if they exist, else replace with defaults"""
        n_dim_state, n_dim_obs = self.n_dim_state, self.n_dim_obs

        arguments = get_params(self)
        defaults = {
            'transition_matrices': np.eye(n_dim_state),
            'transition_offsets': np.zeros(n_dim_state),
            'transition_covariance': np.eye(n_dim_state),
            'observation_matrices': np.eye(n_dim_obs, n_dim_state),
            'observation_offsets': np.zeros(n_dim_obs),
            'observation_covariance': np.eye(n_dim_obs),
            'initial_state_mean': np.zeros(n_dim_state),
            'initial_state_covariance': np.eye(n_dim_state),
            'random_state': 0,
            'em_vars': [
                'transition_covariance',
                'observation_covariance',
                'initial_state_mean',
                'initial_state_covariance'
            ],
        }        
        converters = {
            'transition_matrices': array2d,
            'transition_offsets': array1d,
            'transition_covariance': array2d,
            'observation_matrices': array2d,
            'observation_offsets': array1d,
            'observation_covariance': array2d,
            'initial_state_mean': array1d,
            'initial_state_covariance': array2d,
            'random_state': check_random_state,
            'n_dim_state': int,
            'n_dim_obs': int,
            'em_vars': lambda x: x,
        }

        parameters = preprocess_arguments([arguments, defaults], converters)

        return (
            parameters['transition_matrices'],
            parameters['transition_offsets'],
            parameters['transition_covariance'],
            parameters['observation_matrices'],
            parameters['observation_offsets'],
            parameters['observation_covariance'],
            parameters['initial_state_mean'],
            parameters['initial_state_covariance']
        )        

    def _parse_observations(self, obs):
        """Safely convert observations to their expected format"""
        obs = np.ma.atleast_2d(obs)
        if obs.shape[0] == 1 and obs.shape[1] > 1:
            obs = obs.T
        return obs
