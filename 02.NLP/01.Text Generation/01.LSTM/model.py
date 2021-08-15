import tensorflow as tf
import numpy as np


class Inference(object):
    def __init__(self, model, word2id, id2word, vocab_size, temperature=1.0):
        self.model = model
        self.word2id = word2id
        self.id2word = id2word
        self.temperature = temperature
        self.vocab_size = vocab_size

        # Create a mask to prevent "unk" from being generated.
        skip_ids = np.array([self.word2id("unk")])[:, None]
        sparse_mask = tf.SparseTensor(values=[-float('inf')]*len(skip_ids),
                                      indices=skip_ids,
                                      dense_shape=[vocab_size])
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    def generate_one_step(self, inputs, states=None):

        # convert strings to token IDS
        input_ids = tf.convert_to_tensor([[self.word2id(char) for char in inputs]])

        # Run the model.
        # predicted_logits.shape is [batch, char, next_char_logits]
        predicted_logits, states = self.model(inputs=input_ids, states=states,
                                              return_state=True)
        # Only use the last prediction.
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits / self.temperature

        # Apply the prediction mask: prevent "[UNK]" from being generated.
        predicted_logits = predicted_logits + self.prediction_mask

        # Sample the output logits to generate token IDs.
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids)

        # Convert from token ids to characters
        predicted_chars = self.id2word(predicted_ids)

        # Return the characters and model state.
        return predicted_chars, states


class LSTMModel(tf.keras.Model):
    def __init__(self, vocab_size, batch_size=64, num_steps=50,
                 lstm_size=128, num_layers=2, sampling=False, drop_out=0.5,
                 use_embedding=False, embedding_size=128):
        super(LSTMModel, self).__init__()

        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.drop_out = drop_out
        self.use_embedding = use_embedding
        self.embedding_size = embedding_size
        self.sampling = sampling

        # define model layers
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size)
        lstm_cells = [tf.keras.layers.LSTMCell(self.lstm_size, dropout=self.drop_out) for _ in range(self.num_layers)]
        stacked_lstm = tf.keras.layers.StackedRNNCells(lstm_cells)
        self.lstm_layers = tf.keras.layers.RNN(stacked_lstm, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(self.vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.lstm_layers.get_initial_state(x)
        x, states, _ = self.lstm_layers(x, initial_state=states, training=training)
        x = self.dense(x, training=training)
        if return_state:
            return x, states
        else:
            return x




