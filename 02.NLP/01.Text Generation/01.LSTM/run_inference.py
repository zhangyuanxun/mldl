from data_loader import Vocabulary, batch_generator
from model import LSTMModel, Inference
import tensorflow as tf
import time
import os


if __name__ == "__main__":
    # build Vocabulary
    vocabulary = Vocabulary()
    vocabulary.load_vocab("vocab.pkl")

    # Directory where the checkpoints will be saved
    checkpoint_dir = tf.train.latest_checkpoint('./training_checkpoints')

    num_sample = 1000
    batch_size = 128
    seq_len = 20

    model = LSTMModel(vocabulary.vocab_size, batch_size=batch_size, num_steps=seq_len,
                      lstm_size=128, num_layers=2, sampling=False, drop_out=0.5,
                      use_embedding=False, embedding_size=128)

    model.load_weights(checkpoint_dir)

    inference = Inference(model=model, word2id=vocabulary.word2id,
                          id2word=vocabulary.id2word,
                          vocab_size=vocabulary.vocab_size)

    # define hyper parameters
    start = "我知道"
    next_chars = start
    state = None
    results = next_chars
    for i in range(num_sample):
        next_chars, states = inference.generate_one_step(next_chars, None)
        results += next_chars

    print(results)