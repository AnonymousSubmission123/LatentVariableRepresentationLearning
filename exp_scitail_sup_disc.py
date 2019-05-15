from models.sup.entailment import EntailmentNoConcatSimple as EntailmentModel
from models.unsup.discriminative import LSTM as LangModel
from trainers.sup.entailment import EntailmentNoConcatDisc as Trainer
from run.sup.entailment import EntailmentNoConcatDisc as Run

import sys
import os
import json
import lasagne


sys.setrecursionlimit(5000000)


main_dir = sys.argv[1]
out_dir = sys.argv[2]


dataset = 'SciTail/processed_using_ARC_processed_glove_vocab_40000_unk_0.05'

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
num_outputs = 2
learnable_separator = False


entailment_nn_kwargs = {
    'depth': 2,
    'hid_units': 256,
}

lang_nn_kwargs = {
    'depth': 2,
    'hid_units': 512,
}

trainer_kwargs = {'entailment_model': EntailmentModel,
                  'lang_model': LangModel,
                  'h_dim': d_h,
                  'max_length': max_length,
                  'num_outputs': num_outputs,
                  'learnable_separator': learnable_separator,
                  'entailment_nn_kwargs': entailment_nn_kwargs,
                  'lang_nn_kwargs': lang_nn_kwargs,
                  }

train = True

training_iterations = 1000000
training_batch_size = 64

update = lasagne.updates.adam

lr_init = 0.0001
lr_decay_factor = 0.5
lr_decay_freq = 4000
lr_min = 0.00001

save_params_every = 50000

label_test = False

ans_batch_size = 64


if __name__ == '__main__':

    run = Run(trainer=Trainer, trainer_kwargs=trainer_kwargs, main_dir=main_dir, out_dir=out_dir, dataset=dataset,
              vocab=vocab, load_rec_param_dir=load_rec_param_dir, pre_trained=pre_trained,
              load_param_dir=load_param_dir)

    if train:
        run.train(n_iter=training_iterations, batch_size=training_batch_size, update=update, lr_init=lr_init,
                  lr_decay_factor=lr_decay_factor, lr_decay_freq=lr_decay_freq, lr_min=lr_min,
                  save_params_every=save_params_every)

    if label_test:
        run.label_test(batch_size=ans_batch_size)
