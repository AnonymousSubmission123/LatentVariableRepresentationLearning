import numpy as np
import theano.tensor as T
from lasagne.theano_extensions import conv
from lasagne import init, nonlinearities
from lasagne.layers import DenseLayer, Layer, MergeLayer, Conv1DLayer, PadLayer


class TileLayer(Layer):

    def __init__(self, incoming, reps):
        super().__init__(incoming)
        self.reps = reps

    def get_output_shape_for(self, input_shape, **kwargs):
        output_shape = tuple(input_shape[i] * self.reps[i] if input_shape[i] is not None else None
                             for i in range(len(input_shape)))
        return output_shape

    def get_output_for(self, input, **kwargs):
        return T.tile(input, self.reps)


class SwitchLayer(MergeLayer):

    def __init__(self, incomings, orig, repl):
        super().__init__(incomings)
        self.orig = orig
        self.repl = repl

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, input, **kwargs):
        return T.switch(T.eq(input[0], self.orig), self.repl, input[1])


class DilatedConv1DLayer(Conv1DLayer):

    def __init__(self, incoming, num_filters, filter_size, dilation=1, untie_biases=False, W=init.GlorotUniform(),
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify, flip_filters=False,
                 convolution=conv.conv1d_mc0, **kwargs):

        self.dilation = dilation

        pre_pad = (filter_size - 1) * dilation

        filter_size += (filter_size - 1) * (dilation - 1)

        l_pad = PadLayer(incoming, batch_ndim=2, width=[(pre_pad, 0)])

        super(DilatedConv1DLayer, self).__init__(incoming=l_pad, num_filters=num_filters, filter_size=filter_size,
                                                 stride=1, pad=0, untie_biases=untie_biases, W=W, b=b,
                                                 nonlinearity=nonlinearity, flip_filters=flip_filters,
                                                 convolution=convolution, **kwargs)

    def convolve(self, input, **kwargs):

        border_mode = 'half' if self.pad == 'same' else self.pad

        if self.dilation > 1:
            mask = T.zeros(self.get_W_shape())
            mask = T.set_subtensor(mask[:, :, ::-self.dilation], 1)

        else:
            mask = T.ones_like(self.W)

        conved = self.convolution(input, self.W * mask,
                                  self.input_shape, self.get_W_shape(),
                                  subsample=self.stride,
                                  border_mode=border_mode,
                                  filter_flip=self.flip_filters)

        return conved

