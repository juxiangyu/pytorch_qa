import torch
import numpy as np
import random


def load_vocabulary(vocab_path):
    id_to_word = {}
    with open(vocab_path) as f:
        lines = f.readlines()
        for l in lines:
            d = l.rstrip().split('\t')
            if d[0] not in id_to_word:
                id_to_word[d[0]] = d[1]

    return id_to_word


def load_v1_training_answer_data(file_path, id_2_word, label_2_ans_text, neg_sample_num=1000):
    data = []
    with open(file_path) as f:
        lines = f.readlines()
        for l in lines:
            d = l.rstrip().split('\t')
            question = [id_2_word[t] for t in d[0].split(' ')]  # question
            poss = [label_2_ans_text[t] for t in d[1].split(' ')]  # ground-truth
            negs = []

            for idx in range(neg_sample_num):
                sample_id = str(random.randint(1, len(label_2_ans_text)))
                if sample_id not in poss:
                    negs.append(label_2_ans_text[sample_id])

            for pos in poss:
                data.append((question, pos, negs))
    return data


def load_v1_test_answer_data(file_path, id_2_word):
    data = []
    with open(file_path) as f:
        lines = f.readlines()
        for l in lines:
            d = l.rstrip().split('\t')
            poss = [t for t in d[0].split(' ')]  # ground-truth
            question = [id_2_word[t] for t in d[1].split(' ')]  # question
            candidates = [t for t in d[2].split(' ')]
            data.append((question, poss, candidates))
    return data


def load_id_2_answer_data(id_2_answer_path, id_2_word):
    id_2_answers = {}
    id_2_answer_text = {}
    with open(id_2_answer_path) as f:
        lines = f.readlines()
        for l in lines:
            label, answer = l.rstrip().split('\t')
            if label not in id_2_answers:
                id_2_answers[label] = answer
                id_2_answer_text[label] = [id_2_word[t] for t in answer.split(' ')]

    return id_2_answers, id_2_answer_text


def load_training_answer_data(file_path, id_to_word, label_to_ans_text):
    data = []
    with open(file_path) as f:
        lines = f.readlines()
        for l in lines:
            d = l.rstrip().split('\t')
            question = [id_to_word[t] for t in d[1].split(' ')]  # question
            poss = [label_to_ans_text[t] for t in d[2].split(' ')]  # ground-truth
            negs = [label_to_ans_text[t] for t in d[3].split(' ') if t not in d[2]]

            for pos in poss:
                data.append((question, pos, negs))
    return data


def load_test_answer_data(file_path, id_to_word):
    data = []
    with open(file_path) as f:
        lines = f.readlines()
        for l in lines[12:]:
            d = l.rstrip().split('\t')
            q = [id_to_word[t] for t in d[1].split(' ')] # question
            poss = [t for t in d[2].split(' ')] # ground-truth
            candidates = [t for t in d[3].split(' ')] # candidate-pool
            data.append((q, poss, candidates))
    return data


def head_data(data, num):
    number = num
    for item in data:
        print(item)
        number -= 1
        if number <= 0:
            break


def load_word_embedding(word2vec, vocab_size, embedding_size, word_2_id):
    embedding_matrix = np.zeros((vocab_size, embedding_size))
    print('embedding matrix shape = ', embedding_matrix.shape)
    count = 0
    for word, idx in word_2_id.items():
        # words not found in embedding index will be all-zeros.
        if word in word2vec.wv:
            embedding_matrix[idx] = word2vec.wv[word]
            count += 1
    print('Find word = {} in word embedding'.format(count))
    return torch.from_numpy(embedding_matrix).type(torch.FloatTensor)
