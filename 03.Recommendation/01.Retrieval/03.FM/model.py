import tensorflow as tf


class FM(tf.keras.Model):
    def __init__(self, field_dims, embedding_dim):
        super(FM, self).__init__()
        input_dim = sum(field_dims)
        self.emb_first = tf.keras.layers.Embedding(input_dim, 1)
        self.emb_second = tf.keras.layers.Embedding(input_dim, embedding_dim)
        self.linear = tf.keras.layers.Dense(1, use_bias=True, activation='sigmoid', kernel_initializer='glorot_uniform')

    def call(self, X):
        square_of_sum = tf.math.square(tf.math.reduce_sum(self.emb_second(X), 1))
        sum_of_square = tf.math.reduce_sum(tf.math.square(self.emb_second(X)), 1)

        first_order = tf.math.reduce_sum(self.emb_first(X), axis=1)
        second_order = 0.5 * tf.math.reduce_sum(square_of_sum - sum_of_square, axis=1, keepdims=True)
        out = self.linear(first_order + second_order)
        return out

    def embedding_lookup(self, feature_idx):
        first_order = self.emb_first(feature_idx)
        second_order = self.emb_second(feature_idx)
        return first_order, second_order


