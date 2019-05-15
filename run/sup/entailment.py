import os
import time
import pickle
import json
import numpy as np
from lasagne.updates import adam


class Entailment(object):

    def __init__(self, trainer, trainer_kwargs, main_dir, out_dir, dataset, vocab, pre_trained=False,
                 load_param_dir=None):

        self.main_dir = main_dir
        self.out_dir = out_dir
        self.load_param_dir = load_param_dir

        self.trainer_kwargs = trainer_kwargs

        self.max_length = trainer_kwargs['max_length']
        self.num_outputs = trainer_kwargs['num_outputs']
        self.learnable_separator = trainer_kwargs['learnable_separator']

        print('loading data')

        start = time.clock()

        self.data_train, self.data_val, self.data_test = self.load_data(dataset)
        print('data loaded; time taken = ' + str(time.clock() - start) + ' seconds')

        self.vocab = vocab
        self.vocab_words = vocab[0]
        self.emb_matrix = np.array(vocab[1])

        self.trainer_kwargs['emb_matrix'] = self.emb_matrix

        self.trainer = trainer(**self.trainer_kwargs)

        self.pre_trained = pre_trained

        if self.pre_trained:
            self.load_params()

    def load_params(self):

        raise NotImplementedError()

    def load_data(self, dataset):

        folder = '../_datasets/' + dataset

        with open(os.path.join(folder, 'sup_Train.json'), 'r') as f:
            data_train = json.loads(f.read())

        with open(os.path.join(folder, 'sup_Dev.json'), 'r') as f:
            data_val = json.loads(f.read())

        with open(os.path.join(folder, 'sup_Test.json'), 'r') as f:
            data_test = json.loads(f.read())

        return data_train, data_val, data_test

    def get_batch(self, dataset, batch_size):

        if batch_size == 'all':
            batch_size = len(dataset)
        indices = np.random.choice(len(dataset), batch_size, replace=False)

        p = [dataset[i]['premise'] for i in indices]
        h = [dataset[i]['hypothesis'] for i in indices]

        ph = []

        for n in range(batch_size):

            ph_n = p[n] + [self.emb_matrix.shape[0]] + h[n] if self.learnable_separator else p[n] + h[n]
            ph_n += [-1] * (self.max_length - len(ph_n))

            ph.append(ph_n)

        ph = np.array(ph)
        l = np.array([dataset[i]['label'] for i in indices])

        return ph, l


class EntailmentNoConcat(Entailment):

    def load_params(self):

        raise NotImplementedError()

    def get_batch(self, dataset, batch_size):

        if batch_size == 'all':
            batch_size = len(dataset)
        indices = np.random.choice(len(dataset), batch_size, replace=False)

        p = [dataset[i]['premise'] for i in indices]
        p_pad = [p_n + [-1]*(self.max_length-len(p_n)) for p_n in p]

        h = [dataset[i]['hypothesis'] for i in indices]
        h_pad = [h_n + [-1]*(self.max_length-len(h_n)) for h_n in h]

        p_pad = np.array(p_pad)
        h_pad = np.array(h_pad)
        l = np.array([dataset[i]['label'] for i in indices])

        return p_pad, h_pad, l


class EntailmentNoConcatDisc(EntailmentNoConcat):

    def __init__(self, trainer, trainer_kwargs, main_dir, out_dir, dataset, vocab, load_rec_param_dir,
                 pre_trained=False, load_param_dir=None):

        super().__init__(trainer, trainer_kwargs, main_dir, out_dir, dataset, vocab, pre_trained, load_param_dir)

        with open(os.path.join(load_rec_param_dir, 'params.save'), 'rb') as f:
            self.trainer.lang_model.set_param_values(pickle.load(f))

    def load_params(self):

        with open(os.path.join(self.load_param_dir, 'entailment_params.save'), 'rb') as f:
            self.trainer.entailment_model.set_param_values(pickle.load(f))

    def save_params(self, updates):

        with open(os.path.join(self.out_dir, 'entailment_params.save'), 'wb') as f:
            pickle.dump(self.trainer.entailment_model.get_param_values(), f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.out_dir, 'updates.save'), 'wb') as f:
            pickle.dump(updates, f, protocol=pickle.HIGHEST_PROTOCOL)

    def train(self, n_iter, batch_size, update=adam, update_kwargs=None, lr_init=0.0001, lr_decay_factor=None,
              lr_decay_freq=None, lr_min=None, val_freq=1000, val_batch_size=64, label_num_outputs=32,
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

        obj_fn = self.trainer.obj_fn()

        label_fn = self.trainer.label_fn()

        for i in range(n_iter):

            start = time.clock()

            p_train, h_train, l_train = self.get_batch(self.data_train, batch_size)

            if lr_decay_factor is None:
                lr = lr_init
            else:
                lr = max([lr_init * (lr_decay_factor ** np.floor(i / lr_decay_freq)), lr_min])

            log_p_l = optimiser_fn(p_train, h_train, l_train, lr)

            print('Iteration ' + str(i+1) + ': objective = ' + str(log_p_l) + ' (time taken = ' +
                  str(time.clock() - start) + ' seconds)')

            if val_freq is not None and i % val_freq == 0:

                p_val, h_val, l_val = self.get_batch(self.data_val, val_batch_size)

                log_p_l_val = obj_fn(p_val, h_val, l_val)

                p_eval, h_eval, l_eval = self.get_batch(self.data_val, label_num_outputs)

                l_gen = label_fn(p_eval, h_eval)

                print('Validation: objective = ' + str(log_p_l_val) + ' proportion correct = ' +
                      str(np.sum(l_eval == l_gen) / len(l_eval)))

                self.label_test(label_num_outputs)

            if save_params_every is not None and i % save_params_every == 0 and i > 0:

                self.save_params(updates)

        self.save_params(updates)

    def label_test(self, batch_size):

        label_fn = self.trainer.label_fn()

        p_all, h_all, l_true_all = self.get_batch(self.data_test, batch_size='all')
        l_gen_all = []

        start = time.clock()
        num_batches_done = 0
        total_num_batches = len(list(range(0, len(l_true_all), batch_size)))

        for i in range(0, len(l_true_all), batch_size):

            increment = min((batch_size, len(l_true_all)-i))

            indices_batch = list(range(i, i+increment))

            p_batch = p_all[indices_batch]
            h_batch = h_all[indices_batch]

            l_gen_batch = label_fn(p_batch, h_batch)

            l_gen_all.append(l_gen_batch)

            num_batches_done += 1

            print(str(num_batches_done) + ' of ' + str(total_num_batches) + ' labelled' +
                  ' time taken = ' + str(time.clock() - start) + ' seconds')

        a_gen_all = np.concatenate(l_gen_all, axis=0)

        print('Test: proportion correct = ' + str(np.sum(l_true_all == a_gen_all) / len(l_true_all)))


class EntailmentNoConcatGen(EntailmentNoConcat):

    def __init__(self, trainer, trainer_kwargs, main_dir, out_dir, dataset, vocab, load_rec_param_dir,
                 pre_trained=False, load_param_dir=None):

        super().__init__(trainer, trainer_kwargs, main_dir, out_dir, dataset, vocab, pre_trained, load_param_dir)

        with open(os.path.join(load_rec_param_dir, 'rec_params.save'), 'rb') as f:
            self.trainer.rec_model.set_param_values(pickle.load(f))

    def load_params(self):

        with open(os.path.join(self.load_param_dir, 'gen_params.save'), 'rb') as f:
            self.trainer.gen_model.set_param_values(pickle.load(f))

    def save_params(self, updates):

        with open(os.path.join(self.out_dir, 'gen_params.save'), 'wb') as f:
            pickle.dump(self.trainer.gen_model.get_param_values(), f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.out_dir, 'updates.save'), 'wb') as f:
            pickle.dump(updates, f, protocol=pickle.HIGHEST_PROTOCOL)

    def train(self, n_iter, batch_size, num_samples, update=adam, update_kwargs=None, lr_init=0.0001,
              lr_decay_factor=None, lr_decay_freq=None, lr_min=None, val_freq=1000, val_batch_size=64,
              val_num_samples=8, label_num_outputs=32, label_num_samples=8, save_params_every=None):

        if update_kwargs is None:
            update_kwargs = {}

        if self.pre_trained:
            with open(os.path.join(self.load_param_dir, 'updates.save'), 'rb') as f:
                saved_update = pickle.load(f)
            np.random.seed()
        else:
            saved_update = None

        optimiser_fn, updates = self.trainer.optimiser_fn(num_samples=num_samples, update=update,
                                                          update_kwargs=update_kwargs, saved_update=saved_update)

        obj_fn = self.trainer.obj_fn(val_num_samples)

        label_fn = self.trainer.label_fn(label_num_samples)

        for i in range(n_iter):

            start = time.clock()

            p_train, h_train, l_train = self.get_batch(self.data_train, batch_size)

            if lr_decay_factor is None:
                lr = lr_init
            else:
                lr = max([lr_init * (lr_decay_factor ** np.floor(i / lr_decay_freq)), lr_min])

            log_p_l = optimiser_fn(p_train, h_train, l_train, lr)

            print('Iteration ' + str(i+1) + ': objective = ' + str(log_p_l) + ' (time taken = ' +
                  str(time.clock() - start) + ' seconds)')

            if val_freq is not None and i % val_freq == 0:
                # print(self.trainer.params[0].get_value()[:10])

                p_val, h_val, l_val = self.get_batch(self.data_val, val_batch_size)

                log_p_l_val = obj_fn(p_val, h_val, l_val)

                p_eval, h_eval, l_eval = self.get_batch(self.data_val, label_num_outputs)

                l_gen, _ = label_fn(p_eval, h_eval)

                print('Validation: objective = ' + str(log_p_l_val) + ' proportion correct = ' +
                      str(np.sum(l_eval == l_gen) / len(l_eval)))

                self.label_test(label_num_outputs, label_num_samples)

            if save_params_every is not None and i % save_params_every == 0 and i > 0:

                self.save_params(updates)

        self.save_params(updates)

    def label_test(self, batch_size, num_samples, means_only=False):

        label_fn = self.trainer.label_fn(num_samples, means_only=means_only)

        p_all, h_all, l_true_all = self.get_batch(self.data_test, batch_size='all')
        l_gen_all = []
        probs_gen_all = []

        start = time.clock()
        num_batches_done = 0
        total_num_batches = len(list(range(0, len(l_true_all), batch_size)))

        for i in range(0, len(l_true_all), batch_size):

            increment = min((batch_size, len(l_true_all)-i))

            indices_batch = list(range(i, i+increment))

            p_batch = p_all[indices_batch]
            h_batch = h_all[indices_batch]

            l_gen_batch, probs_gen_batch = label_fn(p_batch, h_batch)

            l_gen_all.append(l_gen_batch)
            probs_gen_all.append(probs_gen_batch)

            num_batches_done += 1

            print(str(num_batches_done) + ' of ' + str(total_num_batches) + ' labelled' +
                  ' time taken = ' + str(time.clock() - start) + ' seconds')

        l_gen_all = np.concatenate(l_gen_all, axis=0)
        probs_gen_all = np.concatenate(probs_gen_all, axis=0)

        print('Test (means_only = ' + str(means_only) + ') : proportion correct = ' +
              str(np.sum(l_true_all == l_gen_all) / len(l_true_all)))

        np.save(os.path.join(self.out_dir, 'p_all.npy'), p_all)
        np.save(os.path.join(self.out_dir, 'h_all.npy'), h_all)
        np.save(os.path.join(self.out_dir, 'l_true_all.npy'), l_true_all)
        np.save(os.path.join(self.out_dir, 'l_gen_all.npy'), l_gen_all)
        np.save(os.path.join(self.out_dir, 'probs_gen_all.npy'), probs_gen_all)
