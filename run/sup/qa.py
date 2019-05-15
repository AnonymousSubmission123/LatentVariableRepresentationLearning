import os
import time
import pickle
import json
import numpy as np
from lasagne.updates import adam


class QA(object):

    def __init__(self, trainer, trainer_kwargs, main_dir, out_dir, dataset, vocab, pre_trained=False,
                 load_param_dir=None):

        self.main_dir = main_dir
        self.out_dir = out_dir
        self.load_param_dir = load_param_dir

        self.trainer_kwargs = trainer_kwargs

        self.max_length = trainer_kwargs['max_length']
        self.num_choices = trainer_kwargs['num_choices']
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

        with open(os.path.join(folder, 'sup_arc_Train.json'), 'r') as f:
            data_train = json.loads(f.read())

        with open(os.path.join(folder, 'sup_arc_Dev.json'), 'r') as f:
            data_val = json.loads(f.read())

        with open(os.path.join(folder, 'sup_arc_Test.json'), 'r') as f:
            data_test = json.loads(f.read())

        return data_train, data_val, data_test

    def get_batch(self, dataset, batch_size, level=None):

        if level is None:
            if batch_size == 'all':
                batch_size = len(dataset)
            indices = np.random.choice(len(dataset), batch_size, replace=False)
        else:
            indices_available = [i for i in range(len(dataset)) if dataset[i]['level'] == level]
            if batch_size == 'all':
                batch_size = len(indices_available)
            indices = np.random.choice(indices_available, batch_size, replace=False)

        q = [dataset[i]['question'] for i in indices]
        o = [dataset[i]['choices'] for i in indices]

        qo = []
        o_mask = []

        for n in range(batch_size):

            o_mask_n = [1] * len(o[n]) + [0] * (self.num_choices - len(o[n]))
            o_mask.append(o_mask_n)

            if len(o[n]) < self.num_choices:
                o[n] += [[-1]] * (self.num_choices - len(o[n]))

            qo_n = []

            for j in range(self.num_choices):
                qo_n_j = q[n] + [self.emb_matrix.shape[0]] + o[n][j] if self.learnable_separator else q[n] + o[n][j]
                qo_n_j += [-1] * (self.max_length - len(qo_n_j))
                qo_n.append(qo_n_j)

            qo.append(qo_n)

        qo = np.array(qo)
        o_mask = np.array(o_mask)
        a = np.array([dataset[i]['answer'] for i in indices])

        return qo, o_mask, a


class QADisc(QA):

    def __init__(self, trainer, trainer_kwargs, main_dir, out_dir, dataset, vocab, load_rec_param_dir,
                 pre_trained=False, load_param_dir=None):

        super().__init__(trainer, trainer_kwargs, main_dir, out_dir, dataset, vocab, pre_trained, load_param_dir)

        with open(os.path.join(load_rec_param_dir, 'params.save'), 'rb') as f:
            self.trainer.lang_model.set_param_values(pickle.load(f))

    def load_params(self):

        with open(os.path.join(self.load_param_dir, 'qa_params.save'), 'rb') as f:
            self.trainer.qa_model.set_param_values(pickle.load(f))

    def save_params(self, updates):

        with open(os.path.join(self.out_dir, 'qa_params.save'), 'wb') as f:
            pickle.dump(self.trainer.qa_model.get_param_values(), f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.out_dir, 'updates.save'), 'wb') as f:
            pickle.dump(updates, f, protocol=pickle.HIGHEST_PROTOCOL)

    def train(self, n_iter, batch_size, update=adam, update_kwargs=None, lr_init=0.0001, lr_decay_factor=None,
              lr_decay_freq=None, lr_min=None, val_freq=1000, val_batch_size=32, ans_num_outputs=100,
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

        ans_fn = self.trainer.ans_fn()

        for i in range(n_iter):

            start = time.clock()

            qo_train, o_mask_train, a_train = self.get_batch(self.data_train, batch_size)

            if lr_decay_factor is None:
                lr = lr_init
            else:
                lr = max([lr_init * (lr_decay_factor ** np.floor(i / lr_decay_freq)), lr_min])

            log_p_a = optimiser_fn(qo_train, o_mask_train, a_train, lr)

            print('Iteration ' + str(i+1) + ': objective = ' + str(log_p_a) + ' (time taken = ' +
                  str(time.clock() - start) + ' seconds)')

            if val_freq is not None and i % val_freq == 0:

                qo_val, o_mask_val, a_val = self.get_batch(self.data_val, val_batch_size)

                log_p_a_val = obj_fn(qo_val, o_mask_val, a_val)

                qo_eval, o_mask_eval, a_eval = self.get_batch(self.data_val, ans_num_outputs, level='Challenge')

                a_gen = ans_fn(qo_eval, o_mask_eval)

                print('Validation: objective = ' + str(log_p_a_val) + ' proportion correct = ' +
                      str(np.sum(a_eval == a_gen) / len(a_eval)))

                self.ans_test(ans_num_outputs)

            if save_params_every is not None and i % save_params_every == 0 and i > 0:

                self.save_params(updates)

        self.save_params(updates)

    def ans_test(self, batch_size):

        ans_fn = self.trainer.ans_fn()

        qo_all, o_mask_all, a_true_all = self.get_batch(self.data_test, batch_size='all', level='Challenge')
        a_gen_all = []

        start = time.clock()
        num_batches_done = 0
        total_num_batches = len(list(range(0, len(a_true_all), batch_size)))

        for i in range(0, len(a_true_all), batch_size):

            increment = min((batch_size, len(a_true_all)-i))

            indices_batch = list(range(i, i+increment))

            qo_batch = qo_all[indices_batch]
            o_mask_batch = o_mask_all[indices_batch]

            a_gen_batch = ans_fn(qo_batch, o_mask_batch)

            a_gen_all.append(a_gen_batch)

            num_batches_done += 1

            print(str(num_batches_done) + ' of ' + str(total_num_batches) + ' answered' +
                  ' time taken = ' + str(time.clock() - start) + ' seconds')

        a_gen_all = np.concatenate(a_gen_all, axis=0)

        print('Test: proportion correct = ' + str(np.sum(a_true_all == a_gen_all) / len(a_true_all)))


class QAGen(QA):

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
              val_num_samples=8, ans_num_outputs=32, ans_num_samples=8, save_params_every=None):

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

        ans_fn = self.trainer.ans_fn(ans_num_samples)

        for i in range(n_iter):

            start = time.clock()

            qo_train, o_mask_train, a_train = self.get_batch(self.data_train, batch_size)

            if lr_decay_factor is None:
                lr = lr_init
            else:
                lr = max([lr_init * (lr_decay_factor ** np.floor(i / lr_decay_freq)), lr_min])

            log_p_a = optimiser_fn(qo_train, o_mask_train, a_train, lr)

            print('Iteration ' + str(i+1) + ': objective = ' + str(log_p_a) + ' (time taken = ' +
                  str(time.clock() - start) + ' seconds)')

            if val_freq is not None and i % val_freq == 0:

                qo_val, o_mask_val, a_val = self.get_batch(self.data_val, val_batch_size)

                log_p_a_val = obj_fn(qo_val, o_mask_val, a_val)

                qo_eval, o_mask_eval, a_eval = self.get_batch(self.data_val, ans_num_outputs, level='Challenge')

                a_gen, probs_gen = ans_fn(qo_eval, o_mask_eval)

                print('Validation: objective = ' + str(log_p_a_val) + ' proportion correct = ' +
                      str(np.sum(a_eval == a_gen) / len(a_eval)))

                self.ans_test(ans_num_outputs, ans_num_samples)

            if save_params_every is not None and i % save_params_every == 0 and i > 0:

                self.save_params(updates)

        self.save_params(updates)

    def ans_test(self, batch_size, num_samples, means_only=False):

        ans_fn = self.trainer.ans_fn(num_samples, means_only=means_only)

        qo_all, o_mask_all, a_true_all = self.get_batch(self.data_test, batch_size='all', level='Challenge')
        a_gen_all = []
        probs_gen_all = []

        start = time.clock()
        num_batches_done = 0
        total_num_batches = len(list(range(0, len(a_true_all), batch_size)))

        for i in range(0, len(a_true_all), batch_size):

            increment = min((batch_size, len(a_true_all)-i))

            indices_batch = list(range(i, i+increment))

            qo_batch = qo_all[indices_batch]
            o_mask_batch = o_mask_all[indices_batch]

            a_gen_batch, probs_gen_batch = ans_fn(qo_batch, o_mask_batch)

            a_gen_all.append(a_gen_batch)
            probs_gen_all.append(probs_gen_batch)

            num_batches_done += 1

            print(str(num_batches_done) + ' of ' + str(total_num_batches) + ' answered' +
                  ' time taken = ' + str(time.clock() - start) + ' seconds')

        a_gen_all = np.concatenate(a_gen_all, axis=0)
        probs_gen_all = np.concatenate(probs_gen_all, axis=0)

        print('Test (means_only = ' + str(means_only) + ') : proportion correct = ' +
              str(np.sum(a_true_all == a_gen_all) / len(a_true_all)))

        np.save(os.path.join(self.out_dir, 'qo_all.npy'), qo_all)
        np.save(os.path.join(self.out_dir, 'a_true_all.npy'), a_true_all)
        np.save(os.path.join(self.out_dir, 'a_gen_all.npy'), a_gen_all)
        np.save(os.path.join(self.out_dir, 'probs_gen_all.npy'), probs_gen_all)

