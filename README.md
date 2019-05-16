# LatentVariableRepresentationLearning

## Dependencies:
* Theano
* Lasagne
* NumPy
* SciPy
* NLTK

## Train language model to learn representations
* Run `exp_arc.py` to train a latent variable language model on the ARC corpus.
  - For the limited time-frame model, change line 2 to `import LSTMTFLimited as GenModel` and add for example `look_back: 4`  in `gen_nn_kwargs` to set K=4. 
  - For Gen-RNN, change line 2 to `import LSTM as GenModel`
* Run `exp_arc_disc.py` to train a discriminative language model (Non-LVM) on the ARC corpus.

## Fine tuning latent variable model for downstream tasks
* Run `exp_arc_sup.py` to train the QA model on the AI2 Reasoning Challenge. Change `load_rec_param_dir` to the language model output directory.
* Run `exp_scitail_sup.py` to train the entailment model on the SciTail dataset. Change `load_rec_param_dir` to the language model output directory.

## Fine tuning discriminative model for downstream tasks
* Run `exp_arc_sup_disc.py` to train the QA model on the AI2 Reasoning Challenge. Change `load_rec_param_dir` to the discriminative model output directory.
* Run `exp_scitail_sup_disc.py` to train the entailment model on the SciTail dataset. Change `load_rec_param_dir` to the discriminative model output directory.
