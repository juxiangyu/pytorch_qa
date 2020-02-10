import os
import argparse
from gensim.models.keyedvectors import KeyedVectors

from preprocess import *
from model import *

import torch

if __name__ == '__main__':
    # parse command line argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='training', type=str,
                        help='training, evaluation and simulation mode are supported')
    parser.add_argument('--testset', default='test1', type=str, help='Define testset type: [dev/test1/test2] ')
    parser.add_argument('--nn_type', default='attention', type=str,
                        help='3 type to reduce rnn output: max_pooling, mean_pooling, attention')
    parser.add_argument('--data_path', default='./data', type=str, metavar='PATH',
                        help='insurance input data directory')
    parser.add_argument('--checkpoint_path', default='./checkpoints', type=str, metavar='PATH',
                        help='model saving directory')
    parser.add_argument('--embd_type', default='google', type=str,
                        help='word embedding mode selection [none/google], default set to google embedding')
    parser.add_argument("--embed_path", default="./embedding/GoogleNews-vectors-negative300.bin", type=str,
                        metavar='PATH',
                        help="pre-trained word embedding path (Google news word embedding)")
    parser.add_argument("--model_path", default="./model/demo.model", type=str,
                        metavar='PATH', help="pre-trained QA model for evaluation and simulation")
    parser.add_argument('--data_version', default='V1', type=str, help='insurance QA dataset version')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--n_epochs', type=int, default=5, help='epoch num')
    parser.add_argument('--embd_size', type=int, default=300, help='word embedding size')
    parser.add_argument('--hidden_size', type=int, default=128, help='hidden size of one-directional LSTM')
    parser.add_argument('--max_sent_len', type=int, default=200, help='max sentence length')
    parser.add_argument('--margin', type=float, default=0.2, help='margin for loss function')
    parser.add_argument('--neg_pos_rate', type=int, default=2,
                        help='training negative samples and positive samples rate')
    args = parser.parse_args()

    mode = args.mode
    model_type = args.nn_type
    model_path = args.model_path
    testset_type = args.testset
    data_path = os.path.join(args.data_path, args.data_version)
    checkpoint_path = os.path.join(args.checkpoint_path, model_type)
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    version = args.data_version
    embedding_type = args.embd_type
    embedding_size = args.embd_size
    hidden_size = args.hidden_size
    max_sentence_len = args.max_sent_len
    batch_size = args.batch_size
    margin = args.margin
    epoch_num = args.n_epochs
    neg_pos_rate = args.neg_pos_rate

    print("============================ Parameters ==================================")
    print("-- Model bi-lstm +", model_type)
    print("-- Dataset Version = ", version)
    print("-- Hidden Size = ", hidden_size)
    print("-- Embedding type = ", embedding_type)
    print("-- Embedding size = ", embedding_size)
    print("-- Epoch Num = ", epoch_num)
    print("-- Batch Size =  ", batch_size)
    print("-- Margin = ", margin)
    print("-- Negative / Positive = ", neg_pos_rate)
    print("-- Checkpoint path: ", checkpoint_path)
    print("============================ Parameters ==================================")

    vocabulary_path = os.path.join(data_path, "vocabulary")
    if version == "V1":
        answer_path = os.path.join(data_path, "answers.label.token_idx")
        qa_training_data_path = os.path.join(data_path, "question.train.token_idx.label")
        qa_test_data_path = os.path.join(data_path, "question.{}.label.token_idx.pool".format(testset_type))
    elif version == "V2":
        answer_path = os.path.join(data_path, "InsuranceQA.label2answer.token.encoded")
        qa_training_data_path = os.path.join(data_path,
                                             "InsuranceQA.question.anslabel.token.100.pool.solr.train.encoded")
        qa_test_data_path = os.path.join(data_path,
                                         "InsuranceQA.question.anslabel.token.100.pool.solr.test.encoded")
    else:
        print("[Error]: invalid dataset version setting, please use -h to check legal parameter setting")
        exit(1)

    id_2_word = load_vocabulary(vocabulary_path)
    word_to_id = {w: i for i, w in enumerate(id_2_word.values(), 1)}
    word_to_id['<PAD>'] = 0
    vocabulary_size = len(word_to_id)

    id_2_answers, id_2_answer_text = load_id_2_answer_data(answer_path, id_2_word)
    print("answer size: ", len(id_2_answer_text))

    print("raw vocabulary size: ", len(word_to_id))
    if version == "V1":
        training_data = load_v1_training_answer_data(qa_training_data_path, id_2_word, id_2_answer_text)
        test_data = load_v1_test_answer_data(qa_test_data_path, id_2_word)
        validation_data = test_data[:100]

    if version == "V2":
        training_data = load_training_answer_data(qa_training_data_path, id_2_word, id_2_answer_text)
        test_data = load_test_answer_data(qa_test_data_path, id_2_word)
        validation_data = test_data[:100]

    print("============================ Checking data ==================================")
    print("word to id sample: ")
    head_data(word_to_id, 10)
    print("data sample: ")
    sample_id = random.randint(0, len(test_data))
    print("-- question: ", " ".join(test_data[sample_id][0]))
    print("-- pos: ", " ".join(id_2_answer_text[test_data[sample_id][1][0]]))
    print("-- neg: ", " ".join(id_2_answer_text[test_data[sample_id][2][0]]))
    print("============================ Checking data done =============================")

    word_embedding = None
    if embedding_type == "google" and mode == "training":
        print('[Word Embedding]: load google news word embedding')
        word2vec = KeyedVectors.load_word2vec_format(args.embed_path, binary=True)
        word_embedding = load_word_embedding(word2vec, vocabulary_size, args.embd_size, word_to_id)

    model = QA_LSTM(vocabulary_size, args.embd_size, args.hidden_size, word_embedding, model_type)
    if torch.cuda.is_available():
        model.cuda()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

    if mode == "training":
        train_batch(model, training_data, validation_data, optimizer, word_to_id, id_2_answer_text, max_sentence_len,
                    epoch_num, batch_size, margin, neg_pos_rate, checkpoint_path)
    if mode == "evaluation":
        print("Load pre-trained model at ", model_path)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])

        test(model, test_data, id_2_answer_text, max_sentence_len, word_to_id)

    if mode == "simulation":
        print("============================ Simulation ========================")
        print("Load pre-trained model at ", model_path)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
        # candidate question
        questions = generate_candidate_questions(training_data)
        print("candidate questions: ", questions)

        simulation(model, list(id_2_answer_text.values()), word_to_id, max_sentence_len)
