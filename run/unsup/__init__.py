import os
import time
import pickle
import json
import numpy as np
from lasagne.updates import adam


class Run(object):

    def __init__(self, trainer, trainer_kwargs, main_dir, out_dir, dataset, vocab, pre_trained=False,
                 load_param_dir=None):

        self.main_dir = main_dir
        self.out_dir = out_dir
        self.load_param_dir = load_param_dir

        self.trainer_kwargs = trainer_kwargs

        print('loading data')

        start = time.clock()

        self.data_train, self.data_test = self.load_data(dataset)
        print('data loaded; time taken = ' + str(time.clock() - start) + ' seconds')

        self.vocab = vocab
        self.vocab_words = vocab[0]
        self.emb_matrix = np.array(vocab[1])

        self.trainer_kwargs['eos_ind'] = self.vocab_words.index('<EOS>')
        self.trainer_kwargs['emb_matrix'] = self.emb_matrix

        self.trainer = trainer(**self.trainer_kwargs)

        self.pre_trained = pre_trained

        if self.pre_trained:
            self.load_params()

    def load_data(self, dataset, train_prop=0.9):

        folder = '../_datasets/' + dataset

        with open(os.path.join(folder, 'unsup.json'), 'r') as f:
            data = np.array(json.loads(f.read()))

        np.random.seed(1234)

        training_mask = np.random.rand(len(data)) < train_prop

        return data[training_mask], data[~training_mask]

    def load_params(self):

        raise NotImplementedError()

    def save_params(self, updates):

        raise NotImplementedError()

    def get_batch(self, dataset, batch_size):

        indices = np.random.choice(len(dataset), batch_size)
        batch = dataset[indices]

        return batch


class RunDisc(Run):

    def load_params(self):

        with open(os.path.join(self.load_param_dir, 'params.save'), 'rb') as f:
            self.trainer.model.set_param_values(pickle.load(f))

    def save_params(self, updates):

        with open(os.path.join(self.out_dir, 'params.save'), 'wb') as f:
            pickle.dump(self.trainer.model.get_param_values(), f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.out_dir, 'updates.save'), 'wb') as f:
            pickle.dump(updates, f, protocol=pickle.HIGHEST_PROTOCOL)

    def train(self, n_iter, batch_size, update=adam, update_kwargs=None, val_freq=1000, val_batch_size=32,
              save_params_every=None):

        if update_kwargs is None:
            update_kwargs = {}

        if self.pre_trained:
            with open(os.path.join(self.load_param_dir, 'updates.save'), 'rb') as f:
                saved_update = pickle.load(f)
            np.random.seed()
        else:
            saved_update = None

        optimiser_fn, updates = self.trainer.optimiser_fn(update=update, update_kwargs=update_kwargs,
                                                          saved_update=saved_update)

        log_p_x_fn = self.trainer.log_p_x_fn()

        for i in range(n_iter):

            start = time.clock()

            batch_train = self.get_batch(self.data_train, batch_size)

            log_p_x = optimiser_fn(batch_train)

            print('Iteration ' + str(i+1) + ': objective = ' + str(log_p_x) +
                  ' (time taken = ' + str(time.clock() - start) + ' seconds)')

            if val_freq is not None and i % val_freq == 0:

                batch_test = self.get_batch(self.data_test, val_batch_size)

                log_p_x_test = log_p_x_fn(batch_test)

                print('Validation: objective = ' + str(log_p_x_test))

            if save_params_every is not None and i % save_params_every == 0 and i > 0:

                self.save_params(updates)

        self.save_params(updates)


class RunGen(Run):

    def load_params(self):

        with open(os.path.join(self.load_param_dir, 'gen_params.save'), 'rb') as f:
            self.trainer.gen_model.set_param_values(pickle.load(f))

        with open(os.path.join(self.load_param_dir, 'rec_params.save'), 'rb') as f:
            self.trainer.rec_model.set_param_values(pickle.load(f))

    def save_params(self, updates):

        with open(os.path.join(self.out_dir, 'vocab_embs.save'), 'wb') as f:
            pickle.dump(self.trainer.all_embeddings.get_value(), f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.out_dir, 'gen_params.save'), 'wb') as f:
            pickle.dump(self.trainer.gen_model.get_param_values(), f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.out_dir, 'rec_params.save'), 'wb') as f:
            pickle.dump(self.trainer.rec_model.get_param_values(), f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.out_dir, 'updates.save'), 'wb') as f:
            pickle.dump(updates, f, protocol=pickle.HIGHEST_PROTOCOL)

    def call_optimiser_fn(self, optimiser_fn, batch_train, beta, i):

        start = time.clock()

        elbo, kl, ppl = optimiser_fn(batch_train, beta)

        print('Iteration ' + str(i + 1) +
              ' objective = ' + str(elbo) +
              ' kl = ' + str(kl) +
              ' ppl = ' + str(ppl) +
              ' (time taken = ' + str(time.clock() - start) + ' seconds)'
              )

    def call_elbo_fn(self, elbo_fn, batch_test):

        elbo, kl, ppl = elbo_fn(batch_test)

        print('Validation' +
              ' objective = ' + str(elbo) +
              ' kl = ' + str(kl) +
              ' ppl = ' + str(ppl)
              )

    def print_gen(self, x_gen):

        print('='*10)
        print('Generations')
        print('='*10)

        for n in range(x_gen.shape[0]):

            print(' '.join([self.vocab_words[int(i)] for i in x_gen[n] if i >= 0]))
            print('-'*10)

        print('='*10)

    def print_recon(self, x_true, x_recon):

        print('='*10)
        print('Reconstructions')
        print('='*10)

        for n in range(x_true.shape[0]):

            print(' true: ' + ' '.join([self.vocab_words[int(i)] for i in x_true[n] if i >= 0]))
            print('recon: ' + ' '.join([self.vocab_words[int(i)] for i in x_recon[n] if i >= 0]))
            print('-'*10)

        print('='*10)

    def train(self, n_iter, batch_size, num_samples, warm_up_type=None, warm_up_curve=None, warm_up_n_iter=None,
              update=adam, update_kwargs=None, val_freq=1000, val_batch_size=32, val_num_samples=1, gen_freq=1000,
              gen_num_outputs=5, recon_num_samples=2, beam_size=15, save_params_every=None):

        if update_kwargs is None:
            update_kwargs = {}

        if self.pre_trained:
            with open(os.path.join(self.load_param_dir, 'updates.save'), 'rb') as f:
                saved_update = pickle.load(f)
            np.random.seed()
        else:
            saved_update = None

        if warm_up_type == 'prior':
            prior_annealing = True
        else:
            prior_annealing = False

        optimiser_fn, updates = self.trainer.optimiser_fn(num_samples=num_samples, update=update,
                                                          update_kwargs=update_kwargs, saved_update=saved_update,
                                                          prior_annealing=prior_annealing)

        elbo_fn = self.trainer.elbo_fn(val_num_samples)

        generator_fn = self.trainer.gen_fn(gen_num_outputs, beam_size)
        reconstructor_fn = self.trainer.recon_fn(recon_num_samples, beam_size)

        def compute_beta(iteration, max_iterations, sharpness=1):
            if warm_up_curve == 'linear':
                return iteration / max_iterations
            elif warm_up_curve == 'sigmoid':
                return 1 / (1 + np.exp((-(10/max_iterations)*iteration + 5) * sharpness))

        for i in range(n_iter):

            batch_train = self.get_batch(self.data_train, batch_size)

            beta = 1. if warm_up_curve is None or i > warm_up_n_iter else compute_beta(i, warm_up_n_iter)

            self.call_optimiser_fn(optimiser_fn, batch_train, beta, i)

            if val_freq is not None and i % val_freq == 0:

                batch_test = self.get_batch(self.data_test, val_batch_size)

                self.call_elbo_fn(elbo_fn, batch_test)

            if gen_freq is not None and i % gen_freq == 0:

                x_gen = generator_fn()
                self.print_gen(x_gen)

                batch_recon = self.get_batch(self.data_test, gen_num_outputs)
                x_recon = reconstructor_fn(batch_recon)
                self.print_recon(batch_recon, x_recon)

            if save_params_every is not None and i % save_params_every == 0 and i > 0:

                self.save_params(updates)

        self.save_params(updates)

    def test(self, batch_size_elbo, batch_size_mi, num_samples):

        elbo_fn = self.trainer.elbo_fn(num_samples)

        elbo = 0
        kl = 0
        ppl = 0

        num_data = len(self.data_test)

        start = time.clock()
        num_batches_done = 0
        total_num_batches = len(list(range(0, num_data, batch_size_elbo)))

        for i in range(0, num_data, batch_size_elbo):

            increment = min((batch_size_elbo, num_data - i))

            indices_batch = list(range(i, i+increment))

            batch = self.data_test[indices_batch]

            elbo_batch, kl_batch, ppl_batch = elbo_fn(batch)

            elbo += elbo_batch*increment
            kl += kl_batch*increment
            ppl += ppl_batch*increment

            num_batches_done += 1

            print(str(num_batches_done) + ' of ' + str(total_num_batches) + ' tested;' +
                  ' time taken = ' + str(time.clock() - start) + ' seconds' +
                  ' elbo = ' + str(elbo / (num_batches_done*batch_size_elbo)) +
                  ' kl = ' + str(kl / (num_batches_done*batch_size_elbo)) +
                  ' ppl = ' + str(ppl / (num_batches_done*batch_size_elbo))
                  )

        print('elbo = ' + str(elbo / num_data))
        print('kl = ' + str(kl / num_data))
        print('ppl = ' + str(ppl / num_data))

        del elbo_fn

        mi_fn = self.trainer.mi_fn(num_samples)

        mi = 0

        start = time.clock()
        num_batches_done = 0
        total_num_batches = len(list(range(0, num_data, batch_size_mi)))

        for i in range(0, num_data, batch_size_mi):

            increment = min((batch_size_mi, num_data - i))

            indices_batch = list(range(i, i+increment))

            batch = self.data_test[indices_batch]

            mi_batch = mi_fn(batch)

            mi += mi_batch*increment

            num_batches_done += 1

            print(str(num_batches_done) + ' of ' + str(total_num_batches) + ' tested;' +
                  ' time taken = ' + str(time.clock() - start) + ' seconds' +
                  ' mi = ' + str(mi / (num_batches_done*batch_size_mi))
                  )

        print('mi = ' + str(mi / num_data))


class RunGenLagged(Run):

    def load_params(self):

        with open(os.path.join(self.load_param_dir, 'gen_params.save'), 'rb') as f:
            self.trainer.gen_model.set_param_values(pickle.load(f))

        with open(os.path.join(self.load_param_dir, 'rec_params.save'), 'rb') as f:
            self.trainer.rec_model.set_param_values(pickle.load(f))

    def save_params(self, updates):

        with open(os.path.join(self.out_dir, 'gen_params.save'), 'wb') as f:
            pickle.dump(self.trainer.gen_model.get_param_values(), f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.out_dir, 'rec_params.save'), 'wb') as f:
            pickle.dump(self.trainer.rec_model.get_param_values(), f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.out_dir, 'updates.save'), 'wb') as f:
            pickle.dump(updates, f, protocol=pickle.HIGHEST_PROTOCOL)

    def call_optimiser_fn(self, optimiser_fn, batch_train, i, rec_only=False):

        start = time.clock()

        beta = 1.

        elbo, kl, ppl = optimiser_fn(batch_train, beta)

        iter_str = 'Rec iteration ' if rec_only else 'Iteration '

        print(iter_str + str(i + 1) +
              ' objective = ' + str(elbo) +
              ' kl = ' + str(kl) +
              ' ppl = ' + str(ppl) +
              ' (time taken = ' + str(time.clock() - start) + ' seconds)'
              )

    def call_elbo_fn(self, elbo_fn, batch_test):

        elbo, kl, ppl = elbo_fn(batch_test)

        print('Validation' +
              ' objective = ' + str(elbo) +
              ' kl = ' + str(kl) +
              ' ppl = ' + str(ppl)
              )

    def print_gen(self, x_gen):

        print('='*10)
        print('Generations')
        print('='*10)

        for n in range(x_gen.shape[0]):

            print(' '.join([self.vocab_words[int(i)] for i in x_gen[n] if i >= 0]))
            print('-'*10)

        print('='*10)

    def print_recon(self, x_true, x_recon):

        print('='*10)
        print('Reconstructions')
        print('='*10)

        for n in range(x_true.shape[0]):

            print(' true: ' + ' '.join([self.vocab_words[int(i)] for i in x_true[n] if i >= 0]))
            print('recon: ' + ' '.join([self.vocab_words[int(i)] for i in x_recon[n] if i >= 0]))
            print('-'*10)

        print('='*10)

    def train(self, n_iter, n_iter_rec, agr_until, batch_size, num_samples, update=adam, update_kwargs=None,
              val_freq=1000, val_batch_size=32, val_num_samples=1, gen_freq=1000, gen_num_outputs=5,
              recon_num_samples=2, beam_size=15, save_params_every=None):

        if update_kwargs is None:
            update_kwargs = {}

        optimiser_gen_fn, updates_gen = self.trainer.optimiser_fn(num_samples=num_samples, update=update,
                                                                  update_kwargs=update_kwargs, gen_only=True)

        optimiser_rec_fn, updates_rec = self.trainer.optimiser_fn(num_samples=num_samples, update=update,
                                                                  update_kwargs=update_kwargs, rec_only=True)

        elbo_fn = self.trainer.elbo_fn(val_num_samples)

        generator_fn = self.trainer.gen_fn(gen_num_outputs, beam_size)
        reconstructor_fn = self.trainer.recon_fn(recon_num_samples, beam_size)

        i = 0
        while i < agr_until:

            batch_train = self.get_batch(self.data_train, batch_size)
            self.call_optimiser_fn(optimiser_gen_fn, batch_train, i)

            n_iter_rec_i = n_iter_rec[-1][1]

            for n in n_iter_rec:
                if i < n[0]:
                    n_iter_rec_i = n[1]
                    break

            for j in range(n_iter_rec_i):
                batch_train_j = self.get_batch(self.data_train, batch_size)
                self.call_optimiser_fn(optimiser_rec_fn, batch_train_j, j, rec_only=True)

            if val_freq is not None and i % val_freq == 0:
                batch_test = self.get_batch(self.data_test, val_batch_size)
                self.call_elbo_fn(elbo_fn, batch_test)

            if gen_freq is not None and i % gen_freq == 0:
                x_gen = generator_fn()
                self.print_gen(x_gen)

                batch_recon = self.get_batch(self.data_test, gen_num_outputs)
                x_recon = reconstructor_fn(batch_recon)
                self.print_recon(batch_recon, x_recon)

            i += 1

        if self.pre_trained:
            with open(os.path.join(self.load_param_dir, 'updates.save'), 'rb') as f:
                saved_update = pickle.load(f)
            np.random.seed()
        else:
            saved_update = updates_gen.update(updates_rec)

        optimiser_fn, updates = self.trainer.optimiser_fn(num_samples=num_samples, update=update,
                                                          update_kwargs=update_kwargs, saved_update=saved_update)

        self.save_params(updates)

        while i < n_iter:

            batch_train = self.get_batch(self.data_train, batch_size)
            self.call_optimiser_fn(optimiser_fn, batch_train, i)

            if val_freq is not None and i % val_freq == 0:
                batch_test = self.get_batch(self.data_test, val_batch_size)
                self.call_elbo_fn(elbo_fn, batch_test)

            if gen_freq is not None and i % gen_freq == 0:
                x_gen = generator_fn()
                self.print_gen(x_gen)

                batch_recon = self.get_batch(self.data_test, gen_num_outputs)
                x_recon = reconstructor_fn(batch_recon)
                self.print_recon(batch_recon, x_recon)

            if save_params_every is not None and i % save_params_every == 0 and i > 0:
                self.save_params(updates)

            i += 1

        self.save_params(updates)
