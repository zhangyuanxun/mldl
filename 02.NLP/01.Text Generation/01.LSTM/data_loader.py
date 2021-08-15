import numpy as np
import copy
import time
import tensorflow as tf
import pickle
import collections
import os


def batch_generator(data, batch_size, seq_len):
    data = copy.copy(data)
    batch_steps = batch_size * seq_len
    n_batches = int(len(data) / batch_steps)
    data = data[:batch_size * n_batches]
    data = data.reshape((batch_size, -1))
    while True:
        np.random.shuffle(data)
        for n in range(0, data.shape[1], seq_len):
            x = data[:, n:n + seq_len]
            y = np.zeros_like(x)
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
            yield x, y


class Vocabulary(object):
    def __init__(self):
        self.vocab = []

    def load_vocab(self, vocab_file):
        assert os.path.exists(vocab_file)

        with open(vocab_file, 'rb') as f:
            self.vocab = pickle.load(f)

    def build_vocab(self, data, limits=5000):
        counter = collections.Counter(data)
        word_freq = counter.most_common(limits)
        self.vocab, freq = zip(*word_freq)

    @property
    def vocab_size(self):
        return len(self.vocab) + 1

    def word2id(self, word):
        if word not in self.vocab:
            return len(self.vocab)

        return self.vocab.index(word)

    def id2word(self, idx):
        if idx == len(self.vocab):
            return "unk"
        if idx < len(self.vocab):
            return self.vocab[idx]
        else:
            raise Exception("index out of range")

    def encode(self, text):
        arr = []
        for word in text:
            arr.append(self.word2id(word))
        return np.array(arr)

    def decode(self, arr):
        words = []
        for idx in arr:
            words.append(self.id2word(idx))
        return " ".join(words)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.vocab, f)







