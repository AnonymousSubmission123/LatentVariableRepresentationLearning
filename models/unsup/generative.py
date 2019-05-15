import numpy as np
import theano
import theano.tensor as T
from lasagne.layers import get_all_layers, get_all_param_values, get_all_params, get_output, set_all_param_values
from lasagne.layers import ConcatLayer, DenseLayer, DropoutLayer, ElemwiseSumLayer, InputLayer, LSTMLayer, \
    PadLayer, ReshapeLayer, SliceLayer, Conv1DLayer, DimshuffleLayer
from nn.layers import TileLayer, DilatedConv1DLayer
from utils import get_log_probs, get_probs


class Text(object):

    def __init__(self, z_dim, dist_z, max_length, vocab_size, emb_dim, embedder):

        self.z_dim = z_dim
        self.dist_z = dist_z()

        self.max_length = max_length
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim

        self.embedder = embedder

        self.nn_in, self.nn_out = self.nn_fn()

        print('Generative model params = ' + '{:,}'.format(sum([np.size(p) for v in self.get_param_values()
                                                                for p in v])))

    def nn_fn(self):

        raise NotImplementedError()

    def kl(self, means_rec, covs_rec):

        kl = -0.5 * T.sum(-covs_rec - means_rec**2 + T.ones_like(means_rec) + T.log(covs_rec),
                          axis=range(1, means_rec.ndim))

        return kl

    def log_p_z(self, z):

        log_p_z = self.dist_z.log_density(z)

        return log_p_z

    def sample_z_prior(self, num_samples):

        z = self.dist_z.get_samples(dims=[1, self.z_dim], num_samples=num_samples)

        return z

    def log_p_x(self, z, x, x_emb, all_embeddings, deterministic=True):

        N = x.shape[0]
        S = T.cast(z.shape[0] / N, 'int32')

        x_rep = T.tile(x, (S, 1))  # (S*N) * L
        x_emb_rep = T.tile(x_emb, (S, 1, 1))  # (S*N) * L * E

        tgt_emb = get_output(self.nn_out, {self.nn_in[0]: z, self.nn_in[1]: x_emb_rep}, deterministic=deterministic)
        # (S*N) * L * E

        probs = get_probs(tgt_emb, x_emb_rep, all_embeddings, mode='true')  # (S*N) * L
        log_p_x = get_log_probs(probs, x_rep)  # (S*N)

        return log_p_x

    def update_active_paths(self, l, scores, active_paths_current, N, beam_size):

        best_scores_l_all = T.max(scores, axis=1)  # N * D

        best_scores_l = T.sort(best_scores_l_all, axis=-1)[:, -beam_size:]  # N * B

        active_words_l = T.argsort(best_scores_l_all, axis=1)[:, -beam_size:]  # N * B

        best_paths_l_all = T.argmax(scores, axis=1)  # N * D

        best_paths_l_inds = best_paths_l_all[T.repeat(T.arange(N), beam_size), active_words_l.flatten()]
        best_paths_l_inds = best_paths_l_inds.reshape((N, beam_size))  # N * B

        best_paths_l = active_paths_current[T.repeat(T.arange(N), beam_size), best_paths_l_inds.flatten()].reshape(
            (N, beam_size, self.max_length))  # N * B * L

        active_paths_new = T.set_subtensor(best_paths_l[:, :, l], active_words_l)

        return best_scores_l, active_paths_new

    def beam_search(self, z, all_embeddings, beam_size):

        N = z.shape[0]

        z_rep = T.repeat(z, beam_size, 0)

        best_scores_0 = T.zeros((N, beam_size))  # N * B
        active_paths_init = -T.ones((N, beam_size, self.max_length))  # N * B * L

        def step_forward(l, best_scores_lm1, active_paths_current):

            active_paths_emb = self.embedder(T.cast(active_paths_current, 'int32'), all_embeddings)  # N * B * L * E
            active_paths_emb = active_paths_emb.reshape((N * beam_size, self.max_length, self.emb_dim))  # (N*B) * L * E

            tgt_emb = get_output(self.nn_out, {self.nn_in[0]: z_rep, self.nn_in[1]: active_paths_emb},
                                 deterministic=True)  # (N*B) * L * E
            tgt_emb = tgt_emb[:, l].reshape((N, beam_size, self.emb_dim))  # N * B * E

            probs = get_probs(tgt_emb, None, all_embeddings)  # N * B * D

            scores = T.shape_padright(best_scores_lm1) + T.log(probs)  # N * B * D

            best_scores_l, active_paths_new = self.update_active_paths(l, scores, active_paths_current, N, beam_size)

            return best_scores_l, active_paths_new

        ([best_scores, active_paths], _) = theano.scan(step_forward,
                                                       sequences=T.arange(self.max_length),
                                                       outputs_info=[best_scores_0, active_paths_init],
                                                       )
        # L * N * B and L * N * B * L

        active_paths = active_paths[-1]  # N * B * L

        words = active_paths[:, -1]  # N * L

        return T.cast(words, 'int32')

    def beam_search_samples(self, z, all_embeddings, num_samples, beam_size):

        N = T.cast(z.shape[0] / num_samples, 'int32')

        z_rep = T.repeat(z, beam_size, 0)

        log_p_z = self.log_p_z(z)

        best_scores_0 = T.zeros((N, beam_size))  # N * B
        active_paths_init = -T.ones((N, beam_size, self.max_length))  # N * B * L

        def step_forward(l, best_scores_lm1, active_paths_current):

            active_paths_emb = self.embedder(T.cast(active_paths_current, 'int32'), all_embeddings)  # N * B * L * E
            active_paths_emb = active_paths_emb.reshape((N * beam_size, self.max_length, self.emb_dim))  # (N*B) * L * E
            active_paths_emb = T.tile(active_paths_emb, (num_samples, 1, 1))  # (S*N*B) * L * E

            tgt_emb = get_output(self.nn_out, {self.nn_in[0]: z_rep, self.nn_in[1]: active_paths_emb},
                                 deterministic=True)  # (S*N*B) * L *
            # E
            tgt_emb = tgt_emb[:, l].reshape((num_samples*N, beam_size, self.emb_dim))  # (S*N) * B * E

            probs = get_probs(tgt_emb, None, all_embeddings)  # (S*N) * B * D

            scores = T.shape_padright(T.tile(best_scores_lm1, (num_samples, 1))) + T.log(probs) + \
                ((1./self.max_length) * T.shape_padright(log_p_z, 2))  # (S*N) * B * D
            scores = T.mean(T.reshape(scores, (num_samples, N, beam_size, self.vocab_size)), axis=0)  # N * B * D

            best_scores_l, active_paths_new = self.update_active_paths(l, scores, active_paths_current, N, beam_size)

            return best_scores_l, active_paths_new

        ([best_scores, active_paths], _) = theano.scan(step_forward,
                                                       sequences=T.arange(self.max_length),
                                                       outputs_info=[best_scores_0, active_paths_init],
                                                       name='beam_search_samples',
                                                       )
        # L * N * B and L * N * B * L

        active_paths = active_paths[-1]  # N * B * L

        words = active_paths[:, -1]  # N * L

        return T.cast(words, 'int32')

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

    def __init__(self, z_dim, dist_z, max_length, vocab_size, emb_dim, embedder, nn_kwargs):

        self.nn_hid_units = nn_kwargs['hid_units']
        self.nn_word_drop = nn_kwargs['word_drop']
        self.nn_skip = nn_kwargs['skip']

        super().__init__(z_dim, dist_z, max_length, vocab_size, emb_dim, embedder)

    def nn_fn(self):

        l_in_z = InputLayer((None, self.z_dim))
        l_in_x = InputLayer((None, self.max_length, self.emb_dim))

        l_in_z_reshape = ReshapeLayer(l_in_z, ([0], 1, [1]))
        l_in_z_rep = TileLayer(l_in_z_reshape, (1, self.max_length, 1))

        l_x_pre_pad = SliceLayer(PadLayer(l_in_x, [(1, 0), (0, 0)], batch_ndim=1), indices=slice(0, -1), axis=1)
        l_in_x_pre_pad_drop = DropoutLayer(l_x_pre_pad, self.nn_word_drop, shared_axes=(-1,))

        l_concat = ConcatLayer((l_in_z_rep, l_in_x_pre_pad_drop), axis=-1)

        l_h = LSTMLayer(l_concat, num_units=self.nn_hid_units)

        if self.nn_skip:
            l_h = ConcatLayer((l_h, l_in_z_rep), axis=-1)

        l_out = DenseLayer(l_h, num_units=self.emb_dim, num_leading_axes=2, nonlinearity=None)

        return (l_in_z, l_in_x), l_out


class LSTMTFLimited(Text):

    def __init__(self, z_dim, dist_z, max_length, vocab_size, emb_dim, embedder, nn_kwargs):

        self.nn_depth = nn_kwargs['depth']
        self.nn_hid_units = nn_kwargs['hid_units']
        self.nn_look_back = nn_kwargs['look_back']
        self.nn_look_back_depth = nn_kwargs['look_back_depth']

        super().__init__(z_dim, dist_z, max_length, vocab_size, emb_dim, embedder)

    def nn_fn(self):

        l_in_z = InputLayer((None, self.z_dim))
        l_in_x = InputLayer((None, self.max_length, self.emb_dim))

        l_z_rep = TileLayer(ReshapeLayer(l_in_z, ([0], 1, [1])), (1, self.max_length, 1))

        l_h = None

        l_h_all = []

        for h in range(self.nn_depth):

            if h > 0:
                l_in_h = ConcatLayer((l_h, l_z_rep), axis=-1)
            else:
                l_in_h = l_z_rep

            l_h = LSTMLayer(l_in_h, num_units=self.nn_hid_units)
            l_h_all.append(l_h)

        l_h = ElemwiseSumLayer(l_h_all)

        l_x_pre_pad = []

        for l in range(1, self.nn_look_back+1):
            l_x_pre_pad_l = SliceLayer(PadLayer(l_in_x, [(l, 0), (0, 0)], batch_ndim=1), indices=slice(0, -l), axis=1)
            l_x_pre_pad.append(l_x_pre_pad_l)

        l_x_pre_pad = ConcatLayer(l_x_pre_pad, axis=-1)

        for d in range(self.nn_look_back_depth - 1):
            l_x_pre_pad = DenseLayer(l_x_pre_pad, num_units=self.nn_hid_units, num_leading_axes=2)

        l_concat = ConcatLayer([l_h, l_x_pre_pad], axis=-1)

        l_out = DenseLayer(l_concat, num_units=self.emb_dim, nonlinearity=None, num_leading_axes=2)

        return (l_in_z, l_in_x), l_out


class DilatedCNN(Text):

    def __init__(self, z_dim, dist_z, max_length, vocab_size, emb_dim, embedder, nn_kwargs):

        self.nn_dilations = nn_kwargs['dilations']
        self.nn_channels_internal = nn_kwargs['cnn_channels_internal']
        self.nn_channels_external = nn_kwargs['cnn_channels_external']
        self.nn_filter_size = nn_kwargs['filter_size']
        self.nn_word_drop = nn_kwargs['word_drop']

        super().__init__(z_dim, dist_z, max_length, vocab_size, emb_dim, embedder)

    def nn_fn(self):

        l_in_z = InputLayer((None, self.z_dim))
        l_in_x = InputLayer((None, self.max_length, self.emb_dim))

        l_in_z_reshape = ReshapeLayer(l_in_z, ([0], [1], 1,))
        l_in_z_rep = TileLayer(l_in_z_reshape, (1, 1, self.max_length))

        l_x_pre_pad = SliceLayer(PadLayer(l_in_x, [(1, 0), (0, 0)], batch_ndim=1), indices=slice(0, -1), axis=1)
        l_x_pre_pad = DimshuffleLayer(l_x_pre_pad, (0, 2, 1))
        l_x_pre_pad_drop = DropoutLayer(l_x_pre_pad, self.nn_word_drop, shared_axes=(1,))

        l_concat = ConcatLayer((l_in_z_rep, l_x_pre_pad_drop), axis=1)

        l_in_d = Conv1DLayer(l_concat, num_filters=self.nn_channels_external, pad='same', filter_size=1,
                             nonlinearity=None)

        for d in self.nn_dilations:
            l_cnn1 = Conv1DLayer(l_in_d, filter_size=1, num_filters=self.nn_channels_internal)
            l_dcnn = DilatedConv1DLayer(l_cnn1, filter_size=self.nn_filter_size, num_filters=self.nn_channels_internal,
                                        dilation=d)
            l_cnn2 = Conv1DLayer(l_dcnn, filter_size=1, num_filters=self.nn_channels_external)

            l_in_d = ElemwiseSumLayer([l_in_d, l_cnn2])

        l_final = Conv1DLayer(l_in_d, filter_size=1, num_filters=self.emb_dim, nonlinearity=None)

        l_out = DimshuffleLayer(l_final, (0, 2, 1))

        return (l_in_z, l_in_x), l_out
