import torch
import torch.nn as nn
import random
import numpy as np

from torch.autograd import Variable


class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size, pre_embedding, is_train_embd=True):
        super(WordEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        if pre_embedding is not None:
            print('pre embedding weight is set')
            self.embedding.weight = nn.Parameter(pre_embedding, requires_grad=is_train_embd)

    def forward(self, x):
        return self.embedding(x)


class QA_LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, pre_embedding, mode):
        super(QA_LSTM, self).__init__()
        self.word_embedding = WordEmbedding(vocab_size, embedding_size, pre_embedding)
        self.hidden_size = hidden_size
        self.shared_lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True, bidirectional=True)
        self.cos = nn.CosineSimilarity(dim=1)
        self.mode = mode

    def attention(self, a_lstm_output, q_pooling):
        # attention weights (B, L, H) -> (B, L, 1)
        q_pooling = q_pooling.unsqueeze(1)
        # a_lstm_output (B, L, H), q pooling (B, 1, H) = weighits (B, L, 1)
        attention_dot = torch.bmm(a_lstm_output, q_pooling.permute(0, 2, 1))
        attention_weights = torch.nn.functional.softmax(attention_dot, dim=1)
        # attention weights * attention value
        output = attention_weights * a_lstm_output

        return output

    def forward(self, q, a):
        # embedding
        q = self.word_embedding(q)  # (bs, L, E)
        a = self.word_embedding(a)  # (bs, L, E)

        # LSTM
        q, q_h = self.shared_lstm(q)  # (bs, L, 2H)
        a, a_h = self.shared_lstm(a)  # (bs, L, 2H)

        # mean
        if self.mode == "mean_pooling":
            q = torch.mean(q, 1)  # (bs, 2H)
            a = torch.mean(a, 1)  # (bs, 2H)
        # max pooling
        elif self.mode == "max_pooling":
            q = torch.max(q, 1)[0]  # (bs, 2H)
            a = torch.max(a, 1)[0]  # (bs, 2H)
        elif self.mode == "attention":
            q = torch.mean(q, 1)  # (bs, 2H)
            a = self.attention(a, q)
            a = torch.mean(a, 1)

        cos = self.cos(q, a)

        return cos


def padding(data, max_sent_len, pad_token):
    pad_len = max(0, max_sent_len - len(data))
    data += [pad_token] * pad_len
    data = data[:max_sent_len]
    return data


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def to_np(x):
    return x.data.cpu().numpy()


def make_vector(data, word_to_id, seq_len):
    ret_data = [padding([word_to_id[w] for w in d], seq_len, 0) for d in data]
    variable = to_var(torch.LongTensor(ret_data))
    return variable


def get_list_size(array):
    size = []
    while isinstance(array, list):
        size.append(len(array))
        array = array[0]
    return size


def generate_batch(data, batch_size):
    batches = []
    start_margin = 0
    end_margin = batch_size
    while end_margin < len(data):
        batches.append(data[start_margin:end_margin])
        start_margin += batch_size
        end_margin += batch_size
    batches.append(data[start_margin:len(data)])
    print("[Split data batch]: batch num = {}, batch_size = {}, last batch size = {}".format(len(batches), batch_size,
                                                                                             len(data) - start_margin))
    return batches


def generate_input(batch, neg_pos_rate):
    question_matrix = []
    positive_ans_matrix = []
    negative_ans_matrix = []

    for data in batch:
        q, pos, negs = data[0], data[1], data[2]
        # generate question matrix
        for _ in range(neg_pos_rate):
            question_matrix.append(q)
            positive_ans_matrix.append(pos)
            negative_ans_matrix.append(random.choice(negs))

    return question_matrix, positive_ans_matrix, negative_ans_matrix


def train_batch(model, data, test_data, optimizer, word_2_id, id_2_answer_text, max_sentence_len, n_epochs,
                batch_size, margin, neg_pos_rate, checkpoint_path):
    for epoch in range(n_epochs):
        model.train()
        print("============================= Epoch {} =============================".format(epoch))
        # shuffle
        random.shuffle(data)
        for batch_id, batch in enumerate(generate_batch(data, batch_size)):
            q_matrix, pos_matrix, neg_matrix = generate_input(batch, neg_pos_rate)
            q_tensor = make_vector(q_matrix, word_2_id,
                                   min(max_sentence_len, max([len(sample) for sample in q_matrix])))
            pos_tensor = make_vector(pos_matrix, word_2_id,
                                     min(max_sentence_len, max([len(sample) for sample in pos_matrix])))
            neg_tensor = make_vector(neg_matrix, word_2_id,
                                     min(max_sentence_len, max([len(sample) for sample in neg_matrix])))
            # print("question batch: ", q_tensor)
            # print("pos batch: ", pos_tensor)
            # print("neg batch: ", neg_tensor)

            pos_similarities = model(q_tensor, pos_tensor)
            neg_similarities = model(q_tensor, neg_tensor)

            losses = margin - pos_similarities + neg_similarities
            losses = torch.where(losses > 0, losses, torch.zeros(losses.size()).cuda())
            loss = torch.mean(losses)

            # print("pos similarities = ", pos_similarities)
            # print("neg similarities = ", neg_similarities)
            # print("losses = ", losses)
            if batch_id % 10 == 0:
                print("[Epoch {} Iteration {}] loss = {}".format(epoch, batch_id, loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        filename = '{}/Epoch-{}.model'.format(checkpoint_path, epoch)
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, filename)
        test(model, test_data, id_2_answer_text, max_sentence_len, word_2_id)


def test(model, data, id_2_answer_text, max_sentence_len, word_2_id):
    acc, total = 0, 0
    for item in data:
        q = item[0]
        labels = item[1]
        candidates = item[2]
        print("question: ", " ".join(q))

        # prepare answer labels
        ground_truth_idxs = [candidates.index(l) for l in labels if l in candidates]
        candidates = [id_2_answer_text[c] for c in candidates]  # id to text

        prediction, score = predict(q, candidates, model, word_2_id, max_sentence_len)

        print("answer ({}): ".format(round(score, 4)), " ".join(candidates[prediction]))

        if prediction in ground_truth_idxs:
            print('correct')
            acc += 1
        else:
            print('wrong')
        total += 1
    print('[Evaluation Accuracy]: ', 100 * acc / total, '%')


def predict(question, candidates, model, word_2_id, max_sentence_len):
    batch_q = make_vector([question for _ in candidates], word_2_id, len(question))
    max_candidates_len = min(max_sentence_len, max([len(c) for c in candidates]))
    batch_a = make_vector(candidates, word_2_id, max_candidates_len)

    # predict
    scores = model(batch_q, batch_a).data.cpu()
    prediction = int(np.argmax(scores))
    max_score = float(max(scores))

    return prediction, max_score


def simulation(model, candidates, word_2_id, max_sentence_len, batch_size=1000):
    while True:
        try:
            # Get input sentence
            input_sentence = input('question > ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit':
                print("============================ Simulation done ========================")
                break
            # Normalize sentence
            question = [word for word in input_sentence.split(' ')]
            # Evaluate sentence
            answers = []
            scores = []
            batch_num = int(len(candidates) / batch_size)
            for idx in range(batch_num):
                batch_candidates = candidates[idx * batch_size:(idx + 1) * batch_size]
                prediction, score = predict(question, batch_candidates, model, word_2_id, max_sentence_len)
                answers.append(" ".join(batch_candidates[prediction]))
                scores.append(score)

            final_batch = candidates[batch_num * batch_size:]
            final_prediction, final_score = predict(question, final_batch, model, word_2_id, max_sentence_len)
            answers.append(" ".join(final_batch[final_prediction]))
            scores.append(final_score)

            print("answer ({}) > ".format(round(max(scores), 4)), answers[int(np.argmax(scores))])
            print("------------------------------------------------------------------------------")

        except KeyError:
            print("Error: Encountered unknown word.")


def generate_candidate_questions(data, num=5):
    questions = []
    for idx in range(num):
        question = random.choice(data)[0]
        questions.append(" ".join(question))

    return questions
