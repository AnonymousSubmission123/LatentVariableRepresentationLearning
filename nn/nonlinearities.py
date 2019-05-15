import theano.tensor as T
from lasagne.nonlinearities import softplus


def softmax(x):

    e_x = T.exp(x - x.max(axis=-1, keepdims=True))
    out = e_x / e_x.sum(axis=-1, keepdims=True)

    return out


def softplus_safe(x):

    return softplus(x) + 1.e-10
