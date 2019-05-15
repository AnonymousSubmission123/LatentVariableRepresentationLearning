import numpy as np
import theano
import theano.tensor as T


def embedder(x, all_embeddings):

    all_embeddings = T.concatenate([all_embeddings, T.zeros((1, all_embeddings.shape[1]))], axis=0)

    return all_embeddings[x]


class EntailmentNoConcatDisc(object):

    def __init__(self, entailment_model, lang_model, h_dim, max_length, emb_matrix, num_outputs, learnable_separator,
                 entailment_nn_kwargs, lang_nn_kwargs):

        self.h_dim = h_dim
        self.max_length = max_length
        self.vocab_size = emb_matrix.shape[0]
        self.emb_dim = emb_matrix.shape[1]
        self.num_outputs = num_outputs

        self.all_embeddings = theano.shared(np.float32(emb_matrix))

        self.entailment_nn_kwargs = entailment_nn_kwargs
        self.lang_nn_kwargs = lang_nn_kwargs

        self.entailment_model, self.lang_model = self.init_models(entailment_model, lang_model)

        self.params = self.get_params()

    def get_params(self):

        return self.entailment_model.get_params()

    def init_models(self, entailment_model, lang_model):

        entailment_model = entailment_model(self.h_dim, self.num_outputs, self.entailment_nn_kwargs)

        lang_model = lang_model(self.max_length, self.vocab_size, self.emb_dim, self.lang_nn_kwargs)

        return entailment_model, lang_model

    def obj(self, p, h, l, deterministic=True):

        p_emb = embedder(p, self.all_embeddings)
        h_emb = embedder(h, self.all_embeddings)

        h_p = self.lang_model.get_h(p, p_emb)
        h_h = self.lang_model.get_h(h, h_emb)

        l_one_hot = T.extra_ops.to_one_hot(l, self.num_outputs, 'float32')

        p_l = self.entailment_model.p_l(h_p, h_h, l_one_hot, deterministic)

        return T.mean(p_l)

    def obj_fn(self):

        p = T.imatrix('p')
        h = T.imatrix('h')
        l = T.ivector('l')

        p_l = self.obj(p, h, l)

        obj_fn = theano.function(inputs=[p, h, l],
                                 outputs=p_l,
                                 allow_input_downcast=True,
                                 )

        return obj_fn

    def optimiser_fn(self, update, update_kwargs, saved_update=None):

        p = T.imatrix('p')
        h = T.imatrix('h')
        l = T.ivector('l')
        learning_rate = T.scalar('learning_rate')

        p_l = self.obj(p, h, l, deterministic=False)

        grads = T.grad(-p_l, self.params, disconnected_inputs='ignore')

        update_kwargs['loss_or_grads'] = grads
        update_kwargs['params'] = self.params
        update_kwargs['learning_rate'] = learning_rate

        updates = update(**update_kwargs)

        if saved_update is not None:
            for u, v in zip(updates, saved_update.keys()):
                u.set_value(v.get_value())

        optimiser = theano.function(inputs=[p, h, l, learning_rate],
                                    outputs=p_l,
                                    updates=updates,
                                    allow_input_downcast=True,
                                    )

        return optimiser, updates

    def label_fn(self):

        p = T.imatrix('p')
        h = T.imatrix('h')

        p_emb = embedder(p, self.all_embeddings)
        h_emb = embedder(h, self.all_embeddings)

        h_p = self.lang_model.get_h(p, p_emb)
        h_h = self.lang_model.get_h(h, h_emb)

        probs = self.entailment_model.get_probs(h_p, h_h)

        label = T.argmax(probs, axis=1)

        label_fn = theano.function(inputs=[p, h],
                                   outputs=label,
                                   allow_input_downcast=True
                                   )

        return label_fn


class EntailmentNoConcatGen(object):

    def __init__(self, gen_model, rec_model, dist_z, z_dim, max_length, emb_matrix, num_outputs, learnable_separator,
                 gen_nn_kwargs, rec_nn_kwargs):

        self.z_dim = z_dim
        self.max_length = max_length
        self.emb_dim = emb_matrix.shape[1]
        self.num_outputs = num_outputs

        self.dist_z = dist_z

        self.all_embeddings = theano.shared(np.float32(emb_matrix))

        self.gen_nn_kwargs = gen_nn_kwargs
        self.rec_nn_kwargs = rec_nn_kwargs

        self.gen_model, self.rec_model = self.init_models(gen_model, rec_model)

        self.params = self.get_params()

    def get_params(self):

        return self.gen_model.get_params()

    def init_models(self, gen_model, rec_model):

        gen_model = gen_model(self.z_dim, self.num_outputs, self.gen_nn_kwargs)

        rec_model = rec_model(self.z_dim, self.dist_z, self.max_length, self.emb_dim, self.rec_nn_kwargs)

        return gen_model, rec_model

    def obj(self, p, h, l, num_samples, deterministic=True):

        p_emb = embedder(p, self.all_embeddings)
        h_emb = embedder(h, self.all_embeddings)

        z_p = self.rec_model.get_samples(p, p_emb, num_samples)
        z_h = self.rec_model.get_samples(h, h_emb, num_samples)

        l_one_hot = T.extra_ops.to_one_hot(l, self.num_outputs, 'float32')

        p_l = self.gen_model.p_l(z_p, z_h, l_one_hot, deterministic)

        return T.mean(p_l)

    def obj_fn(self, num_samples):

        p = T.imatrix('p')
        h = T.imatrix('h')
        l = T.ivector('l')

        p_l = self.obj(p, h, l, num_samples)

        obj_fn = theano.function(inputs=[p, h, l],
                                 outputs=p_l,
                                 allow_input_downcast=True,
                                 )

        return obj_fn

    def optimiser_fn(self, num_samples, update, update_kwargs, saved_update=None):

        p = T.imatrix('p')
        h = T.imatrix('h')
        l = T.ivector('l')
        learning_rate = T.scalar('learning_rate')

        p_l = self.obj(p, h, l, num_samples, deterministic=False)

        grads = T.grad(-p_l, self.params, disconnected_inputs='ignore')

        update_kwargs['loss_or_grads'] = grads
        update_kwargs['params'] = self.params
        update_kwargs['learning_rate'] = learning_rate

        updates = update(**update_kwargs)

        if saved_update is not None:
            for u, v in zip(updates, saved_update.keys()):
                u.set_value(v.get_value())

        optimiser = theano.function(inputs=[p, h, l, learning_rate],
                                    outputs=p_l,
                                    updates=updates,
                                    allow_input_downcast=True,
                                    )

        return optimiser, updates

    def label_fn(self, num_samples, means_only=False):

        p = T.imatrix('p')
        h = T.imatrix('h')

        N = p.shape[0]

        p_emb = embedder(p, self.all_embeddings)
        h_emb = embedder(h, self.all_embeddings)

        z_p = self.rec_model.get_samples(p, p_emb, num_samples, means_only=means_only)
        z_h = self.rec_model.get_samples(h, h_emb, num_samples, means_only=means_only)

        probs = self.gen_model.get_probs(z_p, z_h)

        probs = probs.reshape((num_samples, N, self.num_outputs))

        label = T.argmax(T.mean(probs, axis=0), axis=-1)

        label_fn = theano.function(inputs=[p, h],
                                   outputs=[label, T.mean(probs, axis=0)],
                                   allow_input_downcast=True
                                   )

        return label_fn

