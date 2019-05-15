import numpy as np
import theano
import theano.tensor as T


def embedder(x, all_embeddings):

    all_embeddings = T.concatenate([all_embeddings, T.zeros((1, all_embeddings.shape[1]))], axis=0)

    return all_embeddings[x]


def cut_off(x, eos_ind):

    def step(x_l, x_lm1):

        x_l = T.switch(T.eq(x_lm1, eos_ind), -1, x_l)
        x_l = T.switch(T.eq(x_lm1, -1), -1, x_l)

        return T.cast(x_l, 'int32')

    x_cut_off, _ = theano.scan(step,
                               sequences=x.T,
                               outputs_info=T.ones((x.shape[0],), 'int32'),
                               )

    return x_cut_off.T


class Discriminative(object):

    def __init__(self, model, max_length, emb_matrix, nn_kwargs, eos_ind):

        self.max_length = max_length
        self.vocab_size = emb_matrix.shape[0]
        self.emb_dim = emb_matrix.shape[1]

        self.all_embeddings = theano.shared(np.float32(emb_matrix))

        self.nn_kwargs = nn_kwargs

        self.model = model(self.max_length, self.vocab_size, self.emb_dim, self.nn_kwargs)

        self.params = self.model.get_params()

    def log_p_x(self, x, deterministic=True):

        x_emb = embedder(x, self.all_embeddings)  # N * L * E

        log_p_x = self.model.log_p_x(x, x_emb, self.all_embeddings, deterministic)

        return T.mean(log_p_x)

    def log_p_x_fn(self):

        x = T.imatrix('x')  # N * L

        log_p_x = self.log_p_x(x)

        log_p_x_fn = theano.function(inputs=[x],
                                     outputs=log_p_x,
                                     allow_input_downcast=True,
                                     )

        return log_p_x_fn

    def optimiser_fn(self, update, update_kwargs, saved_update=None):

        x = T.imatrix('x')  # N * L

        log_p_x = self.log_p_x(x, deterministic=False)

        grads = T.grad(-log_p_x, self.params, disconnected_inputs='ignore')

        update_kwargs['loss_or_grads'] = grads
        update_kwargs['params'] = self.params

        updates = update(**update_kwargs)

        if saved_update is not None:
            for u, v in zip(updates, saved_update.keys()):
                u.set_value(v.get_value())

        optimiser = theano.function(inputs=[x],
                                    outputs=log_p_x,
                                    updates=updates,
                                    allow_input_downcast=True,
                                    )

        return optimiser, updates


class SGVB(object):

    def __init__(self, gen_model, rec_model, dist_z, z_dim, max_length, emb_matrix, gen_nn_kwargs, rec_nn_kwargs,
                 eos_ind, learn_embs=False):

        self.z_dim = z_dim
        self.max_length = max_length
        self.vocab_size = emb_matrix.shape[0]
        self.emb_dim = emb_matrix.shape[1]

        self.dist_z = dist_z

        self.all_embeddings = theano.shared(np.float32(emb_matrix))

        self.gen_nn_kwargs = gen_nn_kwargs
        self.rec_nn_kwargs = rec_nn_kwargs

        self.gen_model, self.rec_model = self.init_models(gen_model, rec_model)

        self.eos_ind = eos_ind

        self.params = self.get_params(learn_embs)

    def get_params(self, learn_embs):

        params = self.gen_model.get_params() + self.rec_model.get_params()

        if learn_embs:
            params += [self.all_embeddings]

        return params

    def init_models(self, gen_model, rec_model):

        gen_model = gen_model(self.z_dim, self.dist_z, self.max_length, self.vocab_size, self.emb_dim, embedder,
                              self.gen_nn_kwargs)

        rec_model = rec_model(self.z_dim, self.dist_z, self.max_length, self.emb_dim, self.rec_nn_kwargs)

        return gen_model, rec_model

    def elbo(self, x, num_samples, beta=None, deterministic=True):

        x_emb = embedder(x, self.all_embeddings)  # N * L * E

        z, means_rec, covs_rec = self.rec_model.get_samples_and_means_and_covs(x, x_emb, num_samples)

        kl = self.gen_model.kl(means_rec, covs_rec)  # N

        log_p_x = self.gen_model.log_p_x(z, x, x_emb, self.all_embeddings, deterministic)

        if beta is None:
            elbo = T.mean(log_p_x) - T.mean(kl)
        else:
            elbo = T.mean(log_p_x) - T.mean(beta * kl)

        avg_num_words = T.mean(T.sum(T.ge(x, 0), axis=-1))
        ppl = 2. ** ((-elbo/T.log(2.))/avg_num_words)

        return elbo, T.mean(kl), ppl

    def elbo_prior_annealing(self, x, num_samples, beta=None, deterministic=True):

        x_emb = embedder(x, self.all_embeddings)  # N * L * E

        z, means_rec, covs_rec = self.rec_model.get_samples_and_means_and_covs(x, x_emb, num_samples)

        kl = self.gen_model.kl(means_rec, covs_rec)  # N

        log_p_x = self.gen_model.log_p_x(z, x, x_emb, self.all_embeddings, deterministic)

        if beta is None:
            elbo = T.mean(log_p_x) - T.mean(kl)
        else:
            log_p_z = self.gen_model.log_p_z(z)
            log_q_z = self.rec_model.log_q_z(z, means_rec, covs_rec, num_samples)

            elbo = T.mean(log_p_x) + T.mean(beta * log_p_z) - T.mean(log_q_z)

        avg_num_words = T.mean(T.sum(T.ge(x, 0), axis=-1))
        ppl = 2. ** ((-elbo/T.log(2.))/avg_num_words)

        return elbo, T.mean(kl), ppl

    def mi(self, x, num_samples):

        N = x.shape[0]

        x_emb = embedder(x, self.all_embeddings)  # N * L * E

        z, means_rec, covs_rec = self.rec_model.get_samples_and_means_and_covs(x, x_emb, num_samples)  # (S*N) * Z

        kl = self.gen_model.kl(means_rec, covs_rec)  # N

        log_q_z = self.rec_model.log_q_z(z, means_rec, covs_rec, num_samples)  # (S*N)
        log_p_z = self.gen_model.log_p_z(z)  # (S*N)

        log_q_agg = (1./N) * T.mean(T.log(T.mean(T.reshape(T.exp(log_q_z), (num_samples, N)), axis=-1)))
        log_p_agg = (1./N) * T.mean(T.log(T.mean(T.reshape(T.exp(log_p_z), (num_samples, N)), axis=-1)))

        mi = T.mean(kl) - (log_q_agg - log_p_agg)

        return mi

    def elbo_and_mi(self, x, num_samples):

        N = x.shape[0]

        x_emb = embedder(x, self.all_embeddings)  # N * L * E

        z, means_rec, covs_rec = self.rec_model.get_samples_and_means_and_covs(x, x_emb, num_samples)

        kl = self.gen_model.kl(means_rec, covs_rec)  # N

        log_p_x = self.gen_model.log_p_x(z, x, x_emb, self.all_embeddings, deterministic=True)

        elbo = T.mean(log_p_x) - T.mean(kl)

        avg_num_words = T.mean(T.sum(T.ge(x, 0), axis=-1))
        ppl = 2. ** ((-elbo/T.log(2.))/avg_num_words)

        log_q_z = self.rec_model.log_q_z(z, means_rec, covs_rec, num_samples)
        log_p_z = self.gen_model.log_p_z(z)

        log_q_agg = (1./N) * T.mean(T.log(T.mean(T.reshape(T.exp(log_q_z), (num_samples, N)), axis=-1)))
        log_p_agg = (1./N) * T.mean(T.log(T.mean(T.reshape(T.exp(log_p_z), (num_samples, N)), axis=-1)))

        mi = T.mean(kl) - (log_q_agg - log_p_agg)

        return elbo, T.mean(kl), ppl, mi

    def elbo_fn(self, num_samples):

        x = T.imatrix('x')  # N * L

        elbo, kl, ppl = self.elbo(x, num_samples)

        elbo_fn = theano.function(inputs=[x],
                                  outputs=[elbo, kl, ppl],
                                  allow_input_downcast=True,
                                  )

        return elbo_fn

    def mi_fn(self, num_samples):

        x = T.imatrix('x')  # N * L

        mi = self.mi(x, num_samples)

        mi_fn = theano.function(inputs=[x],
                                outputs=mi,
                                allow_input_downcast=True,
                                )

        return mi_fn

    def elbo_and_mi_fn(self, num_samples):

        x = T.imatrix('x')  # N * L

        elbo, kl, ppl, mi = self.elbo_and_mi(x, num_samples)

        elbo_and_mi_fn = theano.function(inputs=[x],
                                         outputs=[elbo, kl, ppl, mi],
                                         allow_input_downcast=True,
                                         )

        return elbo_and_mi_fn

    def optimiser_fn(self, num_samples, update, update_kwargs, saved_update=None, prior_annealing=False, gen_only=False,
                     rec_only=False):

        x = T.imatrix('x')  # N * L
        beta = T.scalar('beta')

        if prior_annealing:
            elbo, kl, ppl = self.elbo_prior_annealing(x, num_samples, beta, deterministic=False)
        else:
            elbo, kl, ppl = self.elbo(x, num_samples, beta, deterministic=False)

        if gen_only:
            params = self.gen_model.get_params()
        elif rec_only:
            params = self.rec_model.get_params()
        else:
            params = self.params

        grads = T.grad(-elbo, params, disconnected_inputs='ignore')

        update_kwargs['loss_or_grads'] = grads
        update_kwargs['params'] = params

        updates = update(**update_kwargs)

        if saved_update is not None:
            for u, v in zip(updates, saved_update.keys()):
                u.set_value(v.get_value())

        optimiser = theano.function(inputs=[x, beta],
                                    outputs=[elbo, kl, ppl],
                                    updates=updates,
                                    allow_input_downcast=True,
                                    )

        return optimiser, updates

    def gen_fn(self, num_samples, beam_size):

        z = self.gen_model.sample_z_prior(num_samples)

        x_gen = self.gen_model.beam_search(z, self.all_embeddings, beam_size)

        x_gen = cut_off(x_gen, self.eos_ind)

        gen_fn = theano.function(inputs=[],
                                 outputs=x_gen,
                                 allow_input_downcast=True,
                                 )

        return gen_fn

    def recon_fn(self, num_samples, beam_size):

        x = T.imatrix('x')  # N * L

        x_emb = embedder(x, self.all_embeddings)  # N * L * E

        z = self.rec_model.get_samples(x, x_emb, num_samples)

        x_gen = self.gen_model.beam_search_samples(z, self.all_embeddings, num_samples, beam_size)

        x_gen = cut_off(x_gen, self.eos_ind)

        recon_fn = theano.function(inputs=[x],
                                   outputs=x_gen,
                                   allow_input_downcast=True,
                                   )

        return recon_fn

