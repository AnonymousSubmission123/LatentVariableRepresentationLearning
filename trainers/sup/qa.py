import numpy as np
import theano
import theano.tensor as T


def embedder(x, all_embeddings):

    all_embeddings = T.concatenate([all_embeddings, T.zeros((1, all_embeddings.shape[1]))], axis=0)

    return all_embeddings[x]


class QADisc(object):

    def __init__(self, qa_model, lang_model, h_dim, max_length, emb_matrix, num_choices, learnable_separator,
                 qa_nn_kwargs, lang_nn_kwargs):

        self.h_dim = h_dim
        self.max_length = max_length
        self.vocab_size = emb_matrix.shape[0]
        self.emb_dim = emb_matrix.shape[1]
        self.num_choices = num_choices
        self.learnable_separator = learnable_separator

        self.all_embeddings = theano.shared(np.float32(emb_matrix))

        if learnable_separator:
            self.separator_embedding = theano.shared(np.float32(np.random.rand(self.emb_dim)))
            self.embeddings = T.concatenate([self.all_embeddings, T.shape_padleft(self.separator_embedding)], axis=0)
        else:
            self.embeddings = self.all_embeddings

        self.qa_nn_kwargs = qa_nn_kwargs
        self.lang_nn_kwargs = lang_nn_kwargs

        self.qa_model, self.lang_model = self.init_models(qa_model, lang_model)

        self.params = self.get_params()

    def get_params(self):

        params = self.qa_model.get_params()

        if self.learnable_separator:
            params += [self.separator_embedding]

        return params

    def init_models(self, qa_model, lang_model):

        qa_model = qa_model(self.h_dim, self.num_choices, self.qa_nn_kwargs)

        lang_model = lang_model(self.max_length, self.vocab_size, self.emb_dim, self.lang_nn_kwargs)

        return qa_model, lang_model

    def obj(self, qo, o_mask, a, deterministic=True):

        N = qo.shape[0]

        qo_flat = qo.reshape((N * self.num_choices, self.max_length))

        qo_emb = embedder(qo_flat, self.embeddings)

        h = self.lang_model.get_h(qo_flat, qo_emb)

        h = h.reshape((N, self.num_choices, self.h_dim))

        a_one_hot = T.extra_ops.to_one_hot(a, self.num_choices, 'float32')

        p_a = self.qa_model.p_a(h, o_mask, a_one_hot, deterministic)

        return T.mean(p_a)

    def obj_fn(self):

        qo = T.itensor3('qo')
        o_mask = T.matrix('o_mask')
        a = T.ivector('a')

        p_a = self.obj(qo, o_mask, a)

        obj_fn = theano.function(inputs=[qo, o_mask, a],
                                 outputs=p_a,
                                 allow_input_downcast=True,
                                 )

        return obj_fn

    def optimiser_fn(self, update, update_kwargs, saved_update=None):

        qo = T.itensor3('qo')
        o_mask = T.matrix('o_mask')
        a = T.ivector('a')
        learning_rate = T.scalar('learning_rate')

        p_a = self.obj(qo, o_mask, a, deterministic=False)

        grads = T.grad(-p_a, self.params, disconnected_inputs='ignore')

        update_kwargs['loss_or_grads'] = grads
        update_kwargs['params'] = self.params
        update_kwargs['learning_rate'] = learning_rate

        updates = update(**update_kwargs)

        if saved_update is not None:
            for u, v in zip(updates, saved_update.keys()):
                u.set_value(v.get_value())

        optimiser = theano.function(inputs=[qo, o_mask, a, learning_rate],
                                    outputs=p_a,
                                    updates=updates,
                                    allow_input_downcast=True,
                                    )

        return optimiser, updates

    def ans_fn(self):

        qo = T.itensor3('qo')
        o_mask = T.matrix('o_mask')

        N = qo.shape[0]

        qo_flat = qo.reshape((N * self.num_choices, self.max_length))

        qo_emb = embedder(qo_flat, self.embeddings)

        h = self.lang_model.get_h(qo_flat, qo_emb)

        h = h.reshape((N, self.num_choices, self.h_dim))

        probs = self.qa_model.get_probs(h, o_mask)

        ans = T.argmax(probs, axis=-1)

        ans_fn = theano.function(inputs=[qo, o_mask],
                                 outputs=ans,
                                 allow_input_downcast=True
                                 )

        return ans_fn


class QAGen(object):

    def __init__(self, gen_model, rec_model, dist_z, z_dim, max_length, emb_matrix, num_choices, learnable_separator,
                 gen_nn_kwargs, rec_nn_kwargs):

        self.z_dim = z_dim
        self.max_length = max_length
        self.emb_dim = emb_matrix.shape[1]
        self.num_choices = num_choices
        self.learnable_separator = learnable_separator

        self.dist_z = dist_z

        self.all_embeddings = theano.shared(np.float32(emb_matrix))

        if learnable_separator:
            self.separator_embedding = theano.shared(np.float32(np.random.rand(self.emb_dim)))
            self.embeddings = T.concatenate([self.all_embeddings, T.shape_padleft(self.separator_embedding)], axis=0)
        else:
            self.embeddings = self.all_embeddings

        self.gen_nn_kwargs = gen_nn_kwargs
        self.rec_nn_kwargs = rec_nn_kwargs

        self.gen_model, self.rec_model = self.init_models(gen_model, rec_model)

        self.params = self.get_params()

    def get_params(self):

        params = self.gen_model.get_params()

        if self.learnable_separator:
            params += [self.separator_embedding]

        return params

    def init_models(self, gen_model, rec_model):

        gen_model = gen_model(self.z_dim, self.num_choices, self.gen_nn_kwargs)

        rec_model = rec_model(self.z_dim, self.dist_z, self.max_length, self.emb_dim, self.rec_nn_kwargs)

        return gen_model, rec_model

    def obj(self, qo, o_mask, a, num_samples, deterministic=True):

        N = qo.shape[0]

        qo_flat = qo.reshape((N * self.num_choices, self.max_length))

        qo_emb = embedder(qo_flat, self.embeddings)

        z, _, _ = self.rec_model.get_samples_and_means_and_covs(qo_flat, qo_emb, num_samples)

        z = z.reshape((N*num_samples, self.num_choices, self.z_dim))

        a_one_hot = T.extra_ops.to_one_hot(a, self.num_choices, 'float32')

        p_a = self.gen_model.p_a(z, o_mask, a_one_hot, deterministic)

        return T.mean(p_a)

    def obj_fn(self, num_samples):

        qo = T.itensor3('qo')
        o_mask = T.matrix('o_mask')
        a = T.ivector('a')

        p_a = self.obj(qo, o_mask, a, num_samples)

        obj_fn = theano.function(inputs=[qo, o_mask, a],
                                 outputs=p_a,
                                 allow_input_downcast=True,
                                 )

        return obj_fn

    def optimiser_fn(self, num_samples, update, update_kwargs, saved_update=None):

        qo = T.itensor3('qo')
        o_mask = T.matrix('o_mask')
        a = T.ivector('a')
        learning_rate = T.scalar('learning_rate')

        p_a = self.obj(qo, o_mask, a, num_samples, deterministic=False)

        grads = T.grad(-p_a, self.params, disconnected_inputs='ignore')

        update_kwargs['loss_or_grads'] = grads
        update_kwargs['params'] = self.params
        update_kwargs['learning_rate'] = learning_rate

        updates = update(**update_kwargs)

        if saved_update is not None:
            for u, v in zip(updates, saved_update.keys()):
                u.set_value(v.get_value())

        optimiser = theano.function(inputs=[qo, o_mask, a, learning_rate],
                                    outputs=p_a,
                                    updates=updates,
                                    allow_input_downcast=True,
                                    )

        return optimiser, updates

    def ans_fn(self, num_samples, means_only=False):

        qo = T.itensor3('qo')
        o_mask = T.matrix('o_mask')

        N = qo.shape[0]

        qo_flat = qo.reshape((N * self.num_choices, self.max_length))

        qo_emb = embedder(qo_flat, self.embeddings)

        z, _, _ = self.rec_model.get_samples_and_means_and_covs(qo_flat, qo_emb, num_samples, means_only=means_only)

        z = z.reshape((N*num_samples, self.num_choices, self.z_dim))

        o_mask_rep = T.tile(o_mask, (num_samples, 1))

        probs = self.gen_model.get_probs(z, o_mask_rep)

        probs = probs.reshape((num_samples, N, self.num_choices))

        ans = T.argmax(T.mean(probs, axis=0), axis=-1)

        ans_fn = theano.function(inputs=[qo, o_mask],
                                 outputs=[ans, T.mean(probs, axis=0)],
                                 allow_input_downcast=True
                                 )

        return ans_fn


