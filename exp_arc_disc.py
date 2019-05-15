from models.unsup.discriminative import LSTM as Model
from trainers.unsup import Discriminative as Trainer
from run.unsup import RunDisc as Run

import sys
import os
import json
import lasagne


sys.setrecursionlimit(5000000)


main_dir = sys.argv[1]
out_dir = sys.argv[2]


dataset = 'ARC/processed_glove_vocab_40000_unk_0.05'

pre_trained = False
load_param_dir = 'code_outputs/2018_11_25_17_00_22'


with open(os.path.join(main_dir, '../_datasets', dataset, 'vocab_words.json'), 'r') as f:
    vocab_words = json.loads(f.read())

with open(os.path.join(main_dir, '../_datasets', dataset, 'vocab_embs.json'), 'r') as f:
    vocab_embs = json.loads(f.read())

vocab = [vocab_words, vocab_embs]


max_length = 150

nn_kwargs = {
    'depth': 2,
    'hid_units': 512,
}

trainer_kwargs = {'model': Model,
                  'max_length': max_length,
                  'nn_kwargs': nn_kwargs,
                  }

train = True

training_iterations = 400000
training_batch_size = 64
training_num_samples = 1

update = lasagne.updates.adam
update_kwargs = {'learning_rate': 0.0001}

save_params_every = 50000


if __name__ == '__main__':

    run = Run(trainer=Trainer, trainer_kwargs=trainer_kwargs, main_dir=main_dir, out_dir=out_dir, dataset=dataset,
              vocab=vocab, pre_trained=pre_trained, load_param_dir=load_param_dir)

    if train:
        run.train(n_iter=training_iterations, batch_size=training_batch_size, update=update,
                  update_kwargs=update_kwargs, save_params_every=save_params_every)
