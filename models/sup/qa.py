import numpy as np
import theano.tensor as T
from lasagne import init
from lasagne.layers import get_all_layers, get_all_param_values, get_all_params, get_output, set_all_param_values
from lasagne.layers import DenseLayer, InputLayer, NonlinearityLayer, ReshapeLayer
from nn.layers import SwitchLayer
from nn.nonlinearities import softmax


class QA(object):

    def __init__(self, z_dim, num_choices):

        self.z_dim = z_dim

        self.num_choices = num_choices

        self.nn_in, self.nn_out = self.nn_fn()

    def nn_fn(self):

        raise NotImplementedError()

    def p_a(self, z, o_mask, a, deterministic=True):

        N = a.shape[0]
        S = T.cast(z.shape[0] / N, 'int32')

        o_mask_rep = T.tile(o_mask, (S, 1))

        probs = get_output(self.nn_out, {self.nn_in[0]: z, self.nn_in[1]: o_mask_rep}, deterministic=deterministic)
        probs = T.switch(T.isnan(probs), 0, probs)
        probs = probs.reshape((S, N, self.num_choices))

        a_rep = T.tile(T.shape_padleft(a), (S, 1, 1))

        probs = T.mean(T.sum(a_rep * probs, axis=-1), axis=0)

        return probs

    def get_probs(self, z, o_mask):

        probs = get_output(self.nn_out, {self.nn_in[0]: z, self.nn_in[1]: o_mask}, deterministic=True)

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


class QASimple(QA):

    def __init__(self, z_dim, num_choices, nn_kwargs):

        self.nn_depth = nn_kwargs['depth']
        self.nn_hid_units = nn_kwargs['hid_units']

        super().__init__(z_dim, num_choices)

    def nn_fn(self):

        l_in_z = InputLayer((None, self.num_choices, self.z_dim))
        l_in_mask = InputLayer((None, self.num_choices))

        l_h = l_in_z

        for h in range(self.nn_depth - 1):
            l_h = DenseLayer(l_h, num_units=self.nn_hid_units, b=None, num_leading_axes=2)

        l_out_flat = DenseLayer(l_h, num_units=1, b=None, nonlinearity=None, num_leading_axes=2, W=init.GlorotNormal(1.))

        l_out_pre_softmax = ReshapeLayer(l_out_flat, ([0], [1]))
        l_out_pre_softmax = SwitchLayer((l_in_mask, l_out_pre_softmax), 0, -np.inf)

        l_out = NonlinearityLayer(l_out_pre_softmax, softmax)

        return (l_in_z, l_in_mask), l_out



