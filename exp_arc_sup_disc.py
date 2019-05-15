from models.sup.qa import QASimple as QAModel
from models.unsup.discriminative import LSTM as LangModel
from trainers.sup.qa import QADisc as Trainer
from run.sup.qa import QADisc as Run

import sys
import os
import pickle
import json
import lasagne


sys.setrecursionlimit(5000000)


main_dir = sys.argv[1]
out_dir = sys.argv[2]


dataset = 'ARC/processed_glove_vocab_40000_unk_0.05'

load_rec_param_dir = 'code_outputs/2018_12_31_10_49_34'

pre_trained = False
load_param_dir = ''


with open(os.path.join(main_dir, '../_datasets', dataset, 'vocab_words.json'), 'r') as f:
    vocab_words = json.loads(f.read())

with open(os.path.join(main_dir, '../_datasets', dataset, 'vocab_embs.json'), 'rb') as f:
    vocab_embs = json.loads(f.read())

vocab = [vocab_words, vocab_embs]


d_h = 512
max_length = 150
num_choices = 5
learnable_separator = False


qa_nn_kwargs = {
    'depth': 1,
    'hid_units': None,
}

lang_nn_kwargs = {
    'depth': 2,
    'hid_units': 512,
}

trainer_kwargs = {'qa_model': QAModel,
                  'lang_model': LangModel,
                  'h_dim': d_h,
                  'max_length': max_length,
                  'num_choices': num_choices,
                  'learnable_separator': learnable_separator,
                  'qa_nn_kwargs': qa_nn_kwargs,
                  'lang_nn_kwargs': lang_nn_kwargs,
                  }

train = True

training_iterations = 500000
training_batch_size = 64

update = lasagne.updates.adam

lr_init = 0.0001
lr_decay_factor = 0.5
lr_decay_freq = 4000
lr_min = 0.00001

save_params_every = 50000

ans_test = False

ans_batch_size = 64
ans_num_samples = 4


if __name__ == '__main__':

    run = Run(trainer=Trainer, trainer_kwargs=trainer_kwargs, main_dir=main_dir, out_dir=out_dir, dataset=dataset,
              vocab=vocab, load_rec_param_dir=load_rec_param_dir, pre_trained=pre_trained,
              load_param_dir=load_param_dir)

    if train:
        run.train(n_iter=training_iterations, batch_size=training_batch_size, update=update, lr_init=lr_init,
                  lr_decay_factor=lr_decay_factor, lr_decay_freq=lr_decay_freq, lr_min=lr_min,
                  save_params_every=save_params_every)

    if ans_test:
        run.ans_test(batch_size=ans_batch_size)
