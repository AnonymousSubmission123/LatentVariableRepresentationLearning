from distributions import GaussianDiagonal
from models.sup.entailment import EntailmentNoConcatSimple as GenModel
from models.unsup.recognition import LSTM as RecModel
from trainers.sup.entailment import EntailmentNoConcatGen as Trainer
from run.sup.entailment import EntailmentNoConcatGen as Run

import sys
import os
import json
import lasagne


sys.setrecursionlimit(5000000)


main_dir = sys.argv[1]
out_dir = sys.argv[2]


dataset = 'SciTail/processed_using_ARC_processed_glove_vocab_40000_unk_0.05'

load_rec_param_dir = 'code_outputs/2019_05_08_18_53_35'

pre_trained = False
load_param_dir = 'code_outputs/2018_11_25_17_00_22/scitail_400_6'


with open(os.path.join(main_dir, '../_datasets', dataset, 'vocab_words.json'), 'r') as f:
    vocab_words = json.loads(f.read())

with open(os.path.join(main_dir, '../_datasets', dataset, 'vocab_embs.json'), 'r') as f:
    vocab_embs = json.loads(f.read())

vocab = [vocab_words, vocab_embs]


d_z = 64
max_length = 150
num_outputs = 2
learnable_separator = False


gen_nn_kwargs = {
    'depth': 2,
    'hid_units': 256,
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
                  'num_outputs': num_outputs,
                  'learnable_separator': learnable_separator,
                  'gen_nn_kwargs': gen_nn_kwargs,
                  'rec_nn_kwargs': rec_nn_kwargs,
                  }

train = True

training_iterations = 500000
training_batch_size = 64
training_num_samples = 8
training_val_num_samples = 64

update = lasagne.updates.adam

lr_init = 1.e-4
lr_decay_factor = None
lr_decay_freq = None
lr_min = 1.e-4

save_params_every = 50000

label_test = True

ans_batch_size = 64
ans_num_samples = 512


if __name__ == '__main__':

    run = Run(trainer=Trainer, trainer_kwargs=trainer_kwargs, main_dir=main_dir, out_dir=out_dir, dataset=dataset,
              vocab=vocab, load_rec_param_dir=load_rec_param_dir, pre_trained=pre_trained,
              load_param_dir=load_param_dir)

    if train:
        run.train(n_iter=training_iterations, batch_size=training_batch_size, num_samples=training_num_samples,
                  update=update, lr_init=lr_init, lr_decay_factor=lr_decay_factor, lr_decay_freq=lr_decay_freq,
                  lr_min=lr_min, save_params_every=save_params_every, val_num_samples=training_val_num_samples)

    if label_test:
        run.label_test(batch_size=ans_batch_size, num_samples=ans_num_samples)
