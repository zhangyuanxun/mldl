import tensorflow as tf


class LSTMModel(object):
    def __init__(self, num_classes, batch_size=64, num_steps=50,
                 lstm_size=128, num_layers=2, learning_rate=0.001,
                 grad_clip=5, sampling=False, drop_out=0.5,
                 use_embedding=False, embedding_size=128):

        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.drop_out = drop_out
        self.use_embedding = use_embedding
        self.embedding_size = embedding_size



