# models.py

from sentiment_data import *
from typing import List

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable


import numpy as np
import random
import time
from tqdm import tqdm

torch.manual_seed(42)


def pad_to_length(np_arr, length):
    """
    Forces np_arr to length by either truncation (if longer) or zero-padding (if shorter)
    :param np_arr:
    :param length: Length to pad to
    :return: a new numpy array with the data from np_arr padded to be of length length. If length is less than the
    length of the base array, truncates instead.
    """
    result = np.zeros(length)
    result[0:np_arr.shape[0]] = np_arr
    return result


class DAN(nn.Module):
    def __init__(self, inp, hid, out):
        super(DAN, self).__init__()
        self.V = nn.Linear(inp, hid)
        self.g = nn.ReLU()
        self.W1 = nn.Linear(hid, hid)
        self.W = nn.Linear(hid, out)
        self.softmax = nn.Softmax(dim=0)
        # Initialize weights according to the Xavier Glorot formula
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W.weight)

    # Forward computation. Backward computation is done implicitly (nn.Module already has an implementation of
    # it that you shouldn't need to override)
    def forward(self, x):
        return self.softmax(self.W(self.g(self.W1(self.g(self.V(x))))))


def form_input(x):
    # Average value for all the embeddings
    avg_emb = np.divide(sum(x), 60)
    return torch.from_numpy(avg_emb).float()


# , using dev_exs for development and returning
# predictions on the *blind* test_exs (all test_exs have label 0 as a dummy placeholder value).
def train_evaluate_ffnn(train_exs: List[SentimentExample], dev_exs: List[SentimentExample], test_exs: List[SentimentExample], word_vectors: WordEmbeddings) -> List[SentimentExample]:
    """
    Train a feedforward neural network on the given training examples, using dev_exs for development, and returns
    predictions on the *blind* test examples passed in. Returned predictions should be SentimentExample objects with
    predicted labels and the same sentences as input (but these won't be read by the external code). The code is set
    up to go all the way to test predictions so you have more freedom to handle example processing as you see fit.
    :param train_exs:
    :param dev_exs:
    :param test_exs:
    :param word_vectors:
    :return:
    """
    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 60
    # To get you started off, we'll pad the training input to 60 words to make it a square matrix.
    train_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in train_exs])
    # Also store the sequence lengths -- this could be useful for training LSTMs
    train_seq_lens = np.array([len(ex.indexed_words) for ex in train_exs])
    # Labels
    train_labels_arr = np.array([ex.label for ex in train_exs])

    word_embed_dim = 300
    # RUN Training and Test
    num_classes = 2
    num_epochs = 20

    ffnn = DAN(word_embed_dim, 100, num_classes)
    intial_learning_rate = 0.001
    optimizer = optim.Adam(ffnn.parameters(), lr=intial_learning_rate)

    for epoch in range(0, num_epochs):
        ex_indices = [i for i in range(0, len(train_labels_arr))]
        random.shuffle(ex_indices)
        total_loss = 0.0
        for idx in ex_indices:
            # Get the word embeddings
            sentence_embed = []
            for word_idx in train_mat[idx]:
                word = word_vectors.word_indexer.get_object(word_idx)
                sentence_embed.append(word_vectors.get_embedding(word))
            x = form_input(sentence_embed)
            y = train_labels_arr[idx]
            # Build one-hot representation of y
            y_onehot = torch.zeros(num_classes)
            y_onehot.scatter_(0, torch.from_numpy(np.asarray(y, dtype=np.int64)), 1)
            # Zero out the gradients from the FFNN object. *THIS IS VERY IMPORTANT TO DO BEFORE CALLING BACKWARD()*
            ffnn.zero_grad()
            probs = ffnn.forward(x)
            # Can also use built-in NLLLoss as a shortcut here (takes log probabilities) but we're being explicit here
            loss = torch.neg(torch.log(probs)).dot(y_onehot)
            total_loss += loss
            loss.backward()
            optimizer.step()
        print("Loss on epoch %i: %f" % (epoch, total_loss))
    print("done training")

    # Evaluate on the dev set
    dev_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in dev_exs])
    dev_labels_arr = np.array([ex.label for ex in dev_exs])

    dev_correct = 0
    for idx in range(0, len(dev_mat)):
        # Note that we only feed in the x, not the y, since we're not training. We're also extracting different
        # quantities from the running of the computation graph, namely the probabilities, prediction, and z
        sentence_embed = []
        for word_idx in dev_mat[idx]:
            word = word_vectors.word_indexer.get_object(word_idx)
            sentence_embed.append(word_vectors.get_embedding(word))
        x = form_input(sentence_embed)
        y = dev_labels_arr[idx]
        probs = ffnn.forward(x)
        prediction = torch.argmax(probs)
        if y == prediction:
            dev_correct += 1
    print(repr(dev_correct) + "/" + repr(len(dev_labels_arr)) + " correct after dev")
    print("dev accuracy", dev_correct/len(dev_labels_arr))

    # Evaluate on the test set
    test_predictions = []
    test_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in test_exs])
    for idx in range(0, len(test_mat)):
        sentence_embed = []
        for word_idx in test_mat[idx]:
            word = word_vectors.word_indexer.get_object(word_idx)
            sentence_embed.append(word_vectors.get_embedding(word))
        x = form_input(sentence_embed)
        probs = ffnn.forward(x)
        prediction = torch.argmax(probs)
        test_predictions.append(SentimentExample(test_exs[idx].indexed_words, int(prediction)))

    return test_predictions


# def form_input_fancy(x):
#     return torch.from_numpy(x).float()

class RNN(nn.Module):
    def __init__(self, emb_dim, emb_weights, hid, n_layers, n_classes, batch_size, dropout):
        super(RNN, self).__init__()

        # model hyperparametrs
        self.n_layers = n_layers
        self.hid = hid
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.emb_dim = emb_dim
        self.dropout_val = dropout

        # Embedding layer from pretrained embeddings
        self.embedding = nn.Embedding.from_pretrained(emb_weights, freeze=True)

        self.dropout = nn.Dropout(self.dropout_val)
        # LSTM Layer
        self.lstm = nn.LSTM(input_size=self.emb_dim,
                            hidden_size=self.hid,
                            num_layers=self.n_layers,
                            # bidirectional=True,
                            batch_first=True)

        self.dropout = nn.Dropout(self.dropout_val)
        # Output layer
        self.out = nn.Linear(self.hid, n_classes)
        self.ls = nn.Softmax(dim=1)

    def forward(self, text, seq_len):
        # hidden = self.init_hidden()
        emb_out = self.embedding(text)
        emb_out = self.dropout(emb_out)
        batch_size, max_len = text.size()
        packed_seq = nn.utils.rnn.pack_padded_sequence(emb_out, seq_len, batch_first=True, enforce_sorted=False)
        # lstm_out, _ = self.lstm(emb_out, hidden)
        lstm_out, (hidden, _) = self.lstm(packed_seq)
        # unpacked_lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        # unpacked_lstm_out = self.dropout(unpacked_lstm_out)
        # out = self.out(unpacked_lstm_out[:, -1, :])
        out = self.out(hidden[-1])
        return self.ls(out)


# Analogous to train_ffnn, but trains your fancier model.
def train_evaluate_fancy(train_exs: List[SentimentExample], dev_exs: List[SentimentExample], test_exs: List[SentimentExample], word_vectors: WordEmbeddings) -> List[SentimentExample]:
    seq_max_len = 60
    train_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in train_exs])
    dev_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in dev_exs])
    test_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in test_exs])

    # sequence lengths
    train_seq_lens = np.array([len(ex.indexed_words) for ex in train_exs])
    dev_seq_lens = np.array([len(ex.indexed_words) for ex in dev_exs])
    test_seq_lens = np.array([len(ex.indexed_words) for ex in test_exs])

    # Labels
    train_labels_arr = np.array([ex.label for ex in train_exs])
    dev_labels_arr = np.array([ex.label for ex in dev_exs])
    test_labels_arr = np.array([ex.label for ex in test_exs])

    # Prepare the data for batching
    mini_batch_size = 50

    tensor_train = torch.stack([torch.LongTensor(i) for i in train_mat])
    tensor_train_label = torch.LongTensor(train_labels_arr)
    tensor_train_seq_len = torch.LongTensor(train_seq_lens)
    train_data_loader = DataLoader(TensorDataset(tensor_train, tensor_train_label, tensor_train_seq_len), batch_size=mini_batch_size, shuffle=True)

    tensor_dev = torch.stack([torch.LongTensor(i) for i in dev_mat])
    tensor_dev_label = torch.LongTensor(dev_labels_arr)
    tensor_dev_seq_len = torch.LongTensor(dev_seq_lens)
    dev_data_loader = DataLoader(TensorDataset(tensor_dev, tensor_dev_label, tensor_dev_seq_len), batch_size=len(tensor_dev_label))

    # Vocabulary and pretrained glove vectors for embedding layer
    vocab_len = len(word_vectors.word_indexer)
    emb_dim = 300
    emb_weights = torch.FloatTensor(word_vectors.vectors)
    # Model hyperparameters
    n_epochs = 25
    hid = 100
    n_layers = 2
    n_classes = 2
    dropout = 0.2
    # Initialize model
    rnn = RNN(emb_dim, emb_weights, hid, n_layers, n_classes, mini_batch_size, dropout)
    # optimizer
    initial_learning_rate = 0.001
    optimizer = optim.Adam(rnn.parameters(), lr=initial_learning_rate)
    loss = nn.CrossEntropyLoss()

    # Train
    for epoch in range(0, n_epochs):
        epoch_loss = 0
        epoch_start_time = time.time()
        for train_batch, label, seq_len in train_data_loader:
            rnn.zero_grad()
            probs = rnn.forward(train_batch, seq_len)
            loss_out = loss(probs, label)
            epoch_loss += loss_out
            loss_out.backward()
            optimizer.step()
        print("Loss on epoch %i: %f" % (epoch, epoch_loss))
        print("For epoch ", epoch, "Training took %f seconds" % (time.time() - epoch_start_time))

        # Evaluate after every epoch
        dev_correct = 0
        # Evaluate
        for dev_batch, label, seq_len in dev_data_loader:
            probs = rnn.forward(dev_batch, seq_len)
            y_hat = probs.argmax(1)
            # print(y_hat)
            dev_correct += int(torch.sum(y_hat == label))
        print(repr(dev_correct) + "/" + repr(len(dev_labels_arr)) + " correct after training")
        print("dev accuracy", dev_correct / len(dev_labels_arr))
    print("done training")

    # Test set
    tensor_test = torch.stack([torch.LongTensor(i) for i in test_mat])
    tensor_test_seq_len = torch.LongTensor(dev_seq_lens)

    probs = rnn.forward(tensor_test, tensor_test_seq_len)
    predicted = probs.argmax(1)
    predicted = predicted.numpy()
    predictions = [SentimentExample(test_exs[idx].indexed_words, predicted[idx]) for idx in range(0, len(test_exs))]
    return predictions

















