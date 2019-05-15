import theano.tensor as T
from lasagne.layers import get_all_layers, get_all_param_values, get_all_params, get_output, set_all_param_values
from lasagne.layers import ConcatLayer, DenseLayer, InputLayer
from nn.nonlinearities import softmax


class EntailmentNoConcat(object):

    def __init__(self, z_dim, num_outputs):

        self.z_dim = z_dim

        self.num_outputs = num_outputs

        self.nn_in, self.nn_out = self.nn_fn()

    def nn_fn(self):

        raise NotImplementedError()

    def p_l(self, z_p, z_h, l, deterministic=True):

        N = l.shape[0]
        S = T.cast(z_p.shape[0] / N, 'int32')

        probs = get_output(self.nn_out, {self.nn_in[0]: z_p, self.nn_in[1]: z_h}, deterministic=deterministic)
        probs = T.switch(T.isnan(probs), 0, probs)
        probs = probs.reshape((S, N, self.num_outputs))

        l_rep = T.tile(T.shape_padleft(l), (S, 1, 1))

        probs = T.mean(T.sum(l_rep * probs, axis=-1), axis=0)

        return probs

    def get_probs(self, z_p, z_h):

        probs = get_output(self.nn_out, {self.nn_in[0]: z_p, self.nn_in[1]: z_h}, deterministic=True)

        probs = T.switch(T.isnan(probs), 0, probs)

        return probs

    def get_params(self):

        nn_params = get_all_params(get_all_layers(self.nn_out), trainable=True)

        return nn_params

    def get_param_values(self):

        nn_params_vals = get_all_param_values(get_all_layers(self.nn_out))

        return [nn_params_vals]

    def set_param_values(self, param_values):

        [nn_params_vals] = param_values

        set_all_param_values(get_all_layers(self.nn_out), nn_params_vals)


class EntailmentNoConcatSimple(EntailmentNoConcat):

    def __init__(self, z_dim, num_choices, nn_kwargs):

        self.nn_depth = nn_kwargs['depth']
        self.nn_hid_units = nn_kwargs['hid_units']

        super().__init__(z_dim, num_choices)

    def nn_fn(self):

        l_in_z_p = InputLayer((None, self.z_dim))
        l_in_z_h = InputLayer((None, self.z_dim))

        l_h = ConcatLayer([l_in_z_p, l_in_z_h], axis=-1)

        for h in range(self.nn_depth - 1):
            l_h = DenseLayer(l_h, num_units=self.nn_hid_units, b=None)

        l_out = DenseLayer(l_h, num_units=self.num_outputs, b=None, nonlinearity=softmax)

        return (l_in_z_p, l_in_z_h), l_out
