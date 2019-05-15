import numpy as np
import theano.tensor as T
from lasagne.layers import get_all_layers, get_all_param_values, get_all_params, get_output, set_all_param_values
from lasagne.layers import DenseLayer, ElemwiseSumLayer, InputLayer, LSTMLayer
from utils import get_log_probs, get_probs


class Text(object):

    def __init__(self, max_length, vocab_size, emb_dim):

        self.max_length = max_length
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim

        self.nn_in, self.nn_h, self.nn_out = self.nn_fn()

        print('Generative model params = ' + '{:,}'.format(sum([np.size(p) for p in self.get_param_values()[0]])))

    def nn_fn(self):

        raise NotImplementedError()

    def log_p_x(self, x, x_emb, all_embeddings, deterministic=True):

        N = x.shape[0]

        x_emb_pre_pad = T.concatenate([T.zeros((N, 1, self.emb_dim)), x_emb], axis=1)[:, :-1]  # N * L * E

        tgt_emb = get_output(self.nn_out, x_emb_pre_pad, deterministic=deterministic)  # N * L * E

        probs = get_probs(tgt_emb, x_emb, all_embeddings, mode='true')  # N * L
        log_p_x = get_log_probs(probs, x)  # N

        return log_p_x

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

    def __init__(self, max_length, vocab_size, emb_dim, nn_kwargs):

        self.nn_depth = nn_kwargs['depth']
        self.nn_hid_units = nn_kwargs['hid_units']

        super().__init__(max_length, vocab_size, emb_dim)

    def nn_fn(self):

        l_in_x_pre_pad = InputLayer((None, self.max_length, self.emb_dim))

        l_h = l_in_x_pre_pad

        l_h_all = []

        for h in range(self.nn_depth):

            l_h = LSTMLayer(l_h, num_units=self.nn_hid_units)

            l_h_all.append(l_h)

        l_h = ElemwiseSumLayer(l_h_all)

        l_out = DenseLayer(l_h, num_units=self.emb_dim, num_leading_axes=2, nonlinearity=None)

        return l_in_x_pre_pad, l_h, l_out

    def get_h(self, x, x_emb):

        N = x.shape[0]

        x_emb_pre_pad = T.concatenate([T.zeros((N, 1, self.emb_dim)), x_emb], axis=1)[:, :-1]  # N * L * E

        h = get_output(self.nn_h, x_emb_pre_pad, deterministic=True)  # N * L * H

        return h[:, -1]
