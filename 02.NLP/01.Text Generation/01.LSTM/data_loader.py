import numpy as np
import copy
import time
import tensorflow as tf
import pickle
import collections
import os


class Vocabulary(object):
    def __init__(self):
        self.vocab = ["unk"]

    def load_vocab(self, vocab_file, limits=5000):
        assert os.path.exists(vocab_file)

        with open(vocab_file) as f:
            self.vocab += pickle.load(f)[:limits]

    def build_vocab(self, data, limits):
        counter = collections.Counter(data)
        word_freq = counter.most_common(limits)
        self.vocab, freq = zip(*word_freq)

    @property
    def vocab_size(self):
        return len(self.vocab) + 1

    def word2id(self, word):
        if word not in self.vocab:
            return 0

        return self.vocab.index(word)

    def id2word(self, idx):
        if idx < len(self.vocab):
            return self.vocab[idx]
        else:
            raise Exception("index out of range")

    def encode(self, text):
        arr = []
        for word in text.split():
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







