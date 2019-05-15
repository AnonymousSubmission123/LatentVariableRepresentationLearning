import theano.tensor as T
from nn.nonlinearities import softmax


def get_probs(target_embeddings, x_embedded, all_embeddings, mode='all'):

    N, L, E = target_embeddings.shape

    if mode == 'all':

        probs_denominators = T.dot(target_embeddings, all_embeddings.T)  # N * L * V

        probs = softmax(probs_denominators)  # N * L * V

    elif mode == 'true':

        probs_denominators = T.dot(target_embeddings, all_embeddings.T)  # N * L * V

        probs_numerators = T.sum(x_embedded * target_embeddings, axis=-1)  # N * L

        m = T.max(probs_denominators, axis=-1)
        probs_numerators -= m
        probs_denominators -= T.shape_padright(m)

        probs_denominators_exp = T.exp(probs_denominators)  # N * L * V

        probs = T.exp(probs_numerators) / T.sum(probs_denominators_exp, axis=-1)  # N * L

    else:

        raise Exception("mode must be in ['all', 'true']")

    return probs


def get_log_probs(probs, x):

    probs += T.cast(1.e-5, 'float32')  # N * L

    log_probs = T.sum(T.switch(T.ge(x, 0), T.log(probs), 0), axis=-1)

    return log_probs
