from distributions import GaussianDiagonal
from models.unsup.generative import LSTM as GenModel
from models.unsup.recognition import LSTM as RecModel
from trainers.unsup import SGVB as Trainer
from run.unsup import RunGen as Run

import sys
import os
import json
import lasagne


sys.setrecursionlimit(5000000)


main_dir = sys.argv[1]
out_dir = sys.argv[2]


dataset = 'ARC/processed_glove_vocab_40000_unk_0.05'

pre_trained = True
load_param_dir = 'code_outputs/2018_12_17_14_21_13'


with open(os.path.join(main_dir, '../_datasets', dataset, 'vocab_words.json'), 'r') as f:
    vocab_words = json.loads(f.read())

with open(os.path.join(main_dir, '../_datasets', dataset, 'vocab_embs.json'), 'r') as f:
    vocab_embs = json.loads(f.read())

vocab = [vocab_words, vocab_embs]


d_z = 64
max_length = 150

learn_embs = False


gen_nn_kwargs = {
    'depth': 1,
    'hid_units': 512,
    'word_drop': 0.0,
}

rec_nn_kwargs = {
    'rnn_bidirectional': False,
    'rnn_depth': 1,
    'rnn_hid_units': 512,
    'dense_depth': 2,
    'dense_hid_units': 512,
}

trainer_kwargs = {'gen_model': GenModel,
                  'rec_model': RecModel,
                  'dist_z': GaussianDiagonal,
                  'z_dim': d_z,
                  'max_length': max_length,
                  'gen_nn_kwargs': gen_nn_kwargs,
                  'rec_nn_kwargs': rec_nn_kwargs,
                  'learn_embs': learn_embs,
                  }

train = False

training_iterations = 800000
training_warm_up_type = 'sigmoid'
training_warm_up_iterations = 50000
training_batch_size = 64
training_num_samples = 1

update = lasagne.updates.adam
update_kwargs = {'learning_rate': 0.0001}

save_params_every = 50000


test = True

test_batch_size_elbo = 4
test_batch_size_mi = 64
test_num_samples = 32


if __name__ == '__main__':

    run = Run(trainer=Trainer, trainer_kwargs=trainer_kwargs, main_dir=main_dir, out_dir=out_dir, dataset=dataset,
              vocab=vocab, pre_trained=pre_trained, load_param_dir=load_param_dir)

    if train:
        run.train(n_iter=training_iterations, batch_size=training_batch_size, num_samples=training_num_samples,
                  warm_up_curve=training_warm_up_type, warm_up_n_iter=training_warm_up_iterations, update=update,
                  update_kwargs=update_kwargs, save_params_every=save_params_every)

    if test:
        run.test(batch_size_elbo=test_batch_size_elbo, batch_size_mi=test_batch_size_mi, num_samples=test_num_samples)
