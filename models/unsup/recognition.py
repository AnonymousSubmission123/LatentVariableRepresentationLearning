import numpy as np
import theano.tensor as T
from lasagne.layers import get_all_layers, get_all_param_values, get_all_params, get_output, set_all_param_values
from lasagne.layers import ConcatLayer, DenseLayer, ElemwiseSumLayer, InputLayer, LSTMLayer, SliceLayer
from nn.nonlinearities import softplus_safe

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

random = RandomStreams()


class Text(object):

    def __init__(self, z_dim, dist_z, max_length, emb_dim):

        self.z_dim = z_dim
        self.dist_z = dist_z()

        self.max_length = max_length
        self.emb_dim = emb_dim

        self.nn_in, self.nn_out = self.nn_fn()

        print('Recognition model params = ' + '{:,}'.format(sum([np.size(p) for p in self.get_param_values()[0]])))

    def nn_fn(self):

        raise NotImplementedError()

    def get_samples(self, x, x_emb, num_samples, means_only=False, deterministic=True):

        means = get_output(self.nn_out[0], {self.nn_in[0]: x_emb, self.nn_in[1]: T.ge(x, 0)},
                           deterministic=deterministic)
        covs = get_output(self.nn_out[1], {self.nn_in[0]: x_emb, self.nn_in[1]: T.ge(x, 0)},
                          deterministic=deterministic)

        if means_only:
            samples = T.tile(means, [num_samples] + [1]*(means.ndim - 1))
        else:
            samples = self.dist_z.get_samples(num_samples, [means, covs])

        return samples

    def get_samples_and_means_and_covs(self, x, x_emb, num_samples, means_only=False, deterministic=True):

        means = get_output(self.nn_out[0], {self.nn_in[0]: x_emb, self.nn_in[1]: T.ge(x, 0)},
                           deterministic=deterministic)
        covs = get_output(self.nn_out[1], {self.nn_in[0]: x_emb, self.nn_in[1]: T.ge(x, 0)},
                          deterministic=deterministic)

        if means_only:
            samples = T.tile(means, [num_samples] + [1]*(means.ndim - 1))
        else:
            samples = self.dist_z.get_samples(num_samples, [means, covs])

        return samples, means, covs

    def log_q_z(self, z, means, covs, num_samples):

        means = T.tile(means, [num_samples] + [1]*(means.ndim - 1))
        covs = T.tile(covs, [num_samples] + [1]*(means.ndim - 1))

        log_q_z = self.dist_z.log_density(z, [means, covs])

        return log_q_z

    def get_params(self):

        nn_params = get_all_params(get_all_layers(self.nn_out), trainable=True)

        return nn_params

    def get_param_values(self):

        nn_params_vals = get_all_param_values(get_all_layers(self.nn_out))

        return [nn_params_vals]

    def set_param_values(self, param_values):

        [nn_params_vals] = param_values

        set_all_param_values(get_all_layers(self.nn_out), nn_params_vals)


class LSTM(Text):

    def __init__(self, z_dim, dist_z, max_length, emb_dim, nn_kwargs):

        self.rnn_bidirectional = nn_kwargs['rnn_bidirectional']
        self.rnn_depth = nn_kwargs['rnn_depth']
        self.rnn_hid_units = nn_kwargs['rnn_hid_units']
        self.nn_dense_depth = nn_kwargs['dense_depth']
        self.nn_dense_hid_units = nn_kwargs['dense_hid_units']

        super().__init__(z_dim, dist_z, max_length, emb_dim)

    def nn_fn(self):

        l_in = InputLayer((None, self.max_length, self.emb_dim))
        l_mask = InputLayer((None, self.max_length))

        l_h = l_in

        l_h_all = []

        for h in range(self.rnn_depth):
            if self.rnn_bidirectional:
                l_fwd = LSTMLayer(l_h, num_units=self.rnn_hid_units, mask_input=l_mask)
                l_bwd = LSTMLayer(l_h, num_units=self.rnn_hid_units, mask_input=l_mask, backwards=True)
                l_h = ConcatLayer((l_fwd, l_bwd), axis=-1)
            else:
                l_h = LSTMLayer(l_h, num_units=self.rnn_hid_units, mask_input=l_mask)
            l_h_all.append(l_h)

        l_h = SliceLayer(ElemwiseSumLayer(l_h_all), indices=-1, axis=1)

        for i in range(self.nn_dense_depth):
            l_h = DenseLayer(l_h, num_units=self.nn_dense_hid_units)

        l_mean = DenseLayer(l_h, self.z_dim, nonlinearity=None)
        l_cov = DenseLayer(l_h, self.z_dim, nonlinearity=softplus_safe)

        return (l_in, l_mask), (l_mean, l_cov)

