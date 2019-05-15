import numpy as np
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

random = RandomStreams()


class GaussianDiagonal(object):

    def density(self, obs, params=None):

        if params is not None:
            means = params[0]
            covs = params[1]  # covs should be of the same size as obs, i.e. only diagonal Gaussians are permitted.
        else:
            means = T.zeros_like(obs)
            covs = T.ones_like(obs)

        normalisers = ((2. * np.pi * covs) ** (-0.5))

        densities = T.exp(-((obs - means) ** 2) / (2. * covs))

        return normalisers * densities

    def log_density(self, obs, params=None, sum_trailing_axes='all_but_one'):

        if params is not None:
            means = params[0]
            covs = params[1]  # covs should be of the same size as obs, i.e. only diagonal Gaussians are permitted.
        else:
            means = T.zeros_like(obs)
            covs = T.ones_like(obs)

        ndim = obs.ndim

        if sum_trailing_axes == 'all_but_one':

            log_normalisers = - 0.5 * T.sum(T.log(2. * np.pi * covs), axis=range(1, ndim))

            log_densities = - 0.5 * T.sum(((obs - means)**2) / covs, axis=range(1, ndim))

        elif type(sum_trailing_axes) is int:

            log_normalisers = - 0.5 * T.sum(T.log(2. * np.pi * covs), axis=range(-sum_trailing_axes, 0))

            log_densities = - 0.5 * T.sum(((obs - means)**2) / covs, axis=range(-sum_trailing_axes, 0))

        else:

            log_normalisers = - 0.5 * T.log(2. * np.pi * covs)

            log_densities = - 0.5 * (((obs - means)**2) / covs)

        return log_normalisers + log_densities

    def get_samples(self, num_samples, params=None, dims=None):

        if params is not None:
            means = params[0]
            covs = params[1]
        else:
            means = T.zeros(dims)
            covs = T.ones(dims)

        means = T.tile(means, [num_samples] + [1]*(means.ndim - 1))
        covs = T.tile(covs, [num_samples] + [1]*(covs.ndim - 1))

        e = random.normal(means.shape)

        samples = means + (T.sqrt(covs) * e)

        return samples
