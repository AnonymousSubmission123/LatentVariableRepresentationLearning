import os
import time
import json
import numpy as np
from nltk.tokenize import word_tokenize
from collections import Counter


max_len = 150
vocab_size = 40000
lim_unk = 0.05
emb_dim = 300


out_dir = 'processed_glove_vocab_' + str(vocab_size)

if lim_unk is not None:
    out_dir += '_unk_' + str(lim_unk)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)


folder = 'ARC-V1-Feb2018-2'
levels = ['Easy', 'Challenge']
modes = ['Train', 'Dev', 'Test']
ans_map = ['A', 'B', 'C', 'D']

fs_sup = {mode: {level: os.path.join(folder, 'ARC-'+level, 'ARC-'+level+'-'+mode+'.jsonl') for level in levels}
          for mode in modes}

f_unsup = 'ARC_Corpus.txt'


def load_glove_dict():

    f_in = os.path.join('..', 'GloVe', 'glove.42B.300d.txt')

    out = {}

    with open(f_in, 'r') as f:

        start = time.clock()

        for line in f.readlines():

            line_split = line.split(' ')
            word = line_split[0]
            out[word] = [float(i) for i in line_split[1:]]

            if len(out) % 1000 == 0:
                print(str(len(out)) + ' GloVe words added; time taken = ' + str(time.clock() - start) + ' seconds')

    return out

glove = load_glove_dict()


word_counts_sup = Counter()
word_counts_unsup = Counter()

for mode in modes:

    for level in levels:

        f_sup = fs_sup[mode][level]

        with open(f_sup, 'r') as f:

            lines_f_counted = 0
            start = time.clock()

            for line in f:

                line_dict = json.loads(line)

                question_tokenized = word_tokenize(line_dict['question']['stem'].strip().lower())

                word_counts_sup.update(question_tokenized)

                choices = line_dict['question']['choices']

                for choice in choices:

                    choice_tokenized = word_tokenize(choice['text'].strip().lower())

                    word_counts_sup.update(choice_tokenized)

                lines_f_counted += 1
                if lines_f_counted % 100 == 0:
                    print(str(lines_f_counted) + ' lines ' + str(f_sup) + ' counted; time taken = ' +
                          str(time.clock() - start) + ' seconds')

with open(os.path.join(folder, f_unsup), 'r') as f:

    lines_f_counted = 0
    start = time.clock()

    for line in f:
        line_tokenized = word_tokenize(line.strip().lower())

        word_counts_unsup.update(line_tokenized)

        lines_f_counted += 1
        if lines_f_counted % 10000 == 0:
            print(str(lines_f_counted) + ' lines unsup counted; time taken = ' + str(time.clock() - start) +
                  ' seconds')


vocab = {w: glove[w] for w in word_counts_sup if w in glove}

words_counts_unsup_most_common = word_counts_unsup.most_common()
i = 0

while len(vocab) < vocab_size:
    w = words_counts_unsup_most_common[i]
    if w[0] not in vocab and w[0] in glove:
        vocab[w[0]] = glove[w[0]]
    i += 1

vocab_words = list(vocab.keys())
vocab_embs = [vocab[w] for w in vocab_words]

eos_token = '<EOS>'
unk_token = '<UNK>'
vocab_words = [eos_token, unk_token] + vocab_words
vocab_embs = [list(np.random.randn(emb_dim)), list(np.random.randn(emb_dim))] + vocab_embs

vocab_size_final = len(vocab_words)
print('vocab size = ' + str(vocab_size_final))

with open(os.path.join(out_dir, 'vocab_words_en.json'), 'w', encoding='utf-8') as f:
    f.write(json.dumps(vocab_words))

with open(os.path.join(out_dir, 'vocab_embs.json'), 'w') as f:
    f.write(json.dumps(vocab_embs))

unk_index = vocab_words.index(unk_token)


for mode in modes:

    data_mode = []

    for level in levels:

        f_sup = fs_sup[mode][level]

        with open(f_sup, 'r') as f:

            lines_f_counted = 0
            start = time.clock()

            for line in f:

                line_dict = json.loads(line)

                data_line = {'level': level}

                question_tokenized = word_tokenize(line_dict['question']['stem'].strip().lower())
                question_processed = []

                for word in question_tokenized:
                    try:
                        question_processed.append(vocab_words.index(word))
                    except ValueError:
                        question_processed.append(unk_index)

                data_line['question'] = question_processed

                choices = line_dict['question']['choices']
                choices_processed = []

                for choice in choices:

                    choice_tokenized = word_tokenize(choice['text'].strip().lower())
                    choice_processed = []

                    for word in choice_tokenized:
                        try:
                            choice_processed.append(vocab_words.index(word))
                        except ValueError:
                            choice_processed.append(unk_index)

                    choices_processed.append(choice_processed)

                data_line['choices'] = choices_processed

                answer_key = line_dict['answerKey']
                choice_keys = [choice['label'] for choice in choices]

                data_line['answer'] = choice_keys.index(answer_key)

                data_mode.append(data_line)

                lines_f_counted += 1
                if lines_f_counted % 100 == 0:
                    print(str(lines_f_counted) + ' lines ' + str(f_sup) + ' processed; time taken = ' +
                          str(time.clock() - start) + ' seconds')

    with open(os.path.join(out_dir, 'sup_' + mode + '.json'), 'w') as f:
        f.write(json.dumps(data_mode))


with open(os.path.join(folder, f_unsup), 'r') as f:

    lines_f_counted = 0
    start = time.clock()

    lines_processed = []

    for line in f:
        line_tokenized = word_tokenize(line.strip().lower())
        line_processed = []

        valid_line = True
        num_unk = 0
        line_len = len(line_tokenized)

        if len(line_tokenized) <= max_len - 1:

            for word in line_tokenized:
                try:
                    line_processed.append(vocab_words.index(word))
                except ValueError:
                    line_processed.append(unk_index)
                    num_unk += 1
                    if num_unk/line_len >= lim_unk:
                        valid_line = False
                        break

            if valid_line:

                line_processed.append(vocab_words.index(eos_token))

                if len(line_processed) < max_len:
                    line_processed += [-1] * (max_len - len(line_processed))

                lines_processed.append(line_processed)

        lines_f_counted += 1
        if lines_f_counted % 10000 == 0:
            print(str(lines_f_counted) + ' lines unsup processed; ' + str(len(lines_processed)) +
                  ' lines unsup accepted; ' + 'time taken = ' + str(time.clock() - start) +
                  ' seconds')

with open(os.path.join(out_dir, 'unsup.json'), 'w') as f:
    f.write(json.dumps(lines_processed))
