import os
import time
import json
from nltk.tokenize import word_tokenize


vocab_dir_path = ['..', 'ARC', 'processed_glove_vocab_40000_unk_0.05']
vocab_dir = os.path.join(*vocab_dir_path)

out_dir = 'processed_using_ARC_' + vocab_dir_path[-1]

if not os.path.exists(out_dir):
    os.makedirs(out_dir)


folder = os.path.join('SciTailV1.1', 'tsv_format')
modes = ['Train', 'Dev', 'Test']
ans_map = ['neutral', 'entails']

f_names = {mode: os.path.join(folder, 'scitail_1.0_' + mode.lower() + '.tsv') for mode in modes}


with open(os.path.join(vocab_dir, 'vocab_words_en.json'), 'r') as f:
    vocab_words = json.loads(f.read())

with open(os.path.join(vocab_dir, 'vocab_embs.json'), 'r') as f:
    vocab_embs = json.loads(f.read())

eos_token = '<EOS>'
unk_token = '<UNK>'

unk_index = vocab_words.index(unk_token)

with open(os.path.join(out_dir, 'vocab_words_en.json'), 'w', encoding='utf-8') as f:
    f.write(json.dumps(vocab_words))

with open(os.path.join(out_dir, 'vocab_embs.json'), 'w') as f:
    f.write(json.dumps(vocab_embs))


max_len = 0

for mode in modes:

    data_mode = []

    f_name = f_names[mode]

    with open(f_name, 'r') as f:

        lines_f_processed = 0
        start = time.clock()

        for line in f:

            premise, hypothesis, label = line.split('\t')

            data_line = {}

            premise_tokenized = word_tokenize(premise.strip().lower())
            premise_processed = []

            for word in premise_tokenized:
                try:
                    premise_processed.append(vocab_words.index(word))
                except ValueError:
                    premise_processed.append(unk_index)

            data_line['premise'] = premise_processed

            hypothesis_tokenized = word_tokenize(hypothesis.strip().lower())
            hypothesis_processed = []

            for word in hypothesis_tokenized:
                try:
                    hypothesis_processed.append(vocab_words.index(word))
                except ValueError:
                    hypothesis_processed.append(unk_index)

            data_line['hypothesis'] = hypothesis_processed

            max_len = max((len(premise_processed + hypothesis_processed), max_len))

            data_line['label'] = ans_map.index(label.strip())

            data_mode.append(data_line)

            lines_f_processed += 1
            if lines_f_processed % 100 == 0:
                print(str(lines_f_processed) + ' lines ' + str(f_name) + ' processed; time taken = ' +
                      str(time.clock() - start) + ' seconds; max len = ' + str(max_len))

    with open(os.path.join(out_dir, 'sup_' + mode + '.json'), 'w') as f:
        f.write(json.dumps(data_mode))
