import tensorflow as tf
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


class LogisticRegression(tf.keras.Model):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.W = tf.keras.layers.Dense(1, use_bias=True, kernel_initializer='random_uniform',
                                       activation='sigmoid')

    def call(self, X):
        return self.W(X)


def train_step(X, y):
    with tf.GradientTape() as tape:
        y_hat = model(X)
        loss = loss_fn(y, y_hat)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(y, y_hat)


def test_step(X, y):
    with tf.GradientTape() as tape:
        y_hat = model(X)

    test_accuracy(y, y_hat)


if __name__ == "__main__":
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33,
                                                        random_state=42)

    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    y_train = tf.expand_dims(tf.convert_to_tensor(y_train, dtype=tf.float32), axis=1)
    print(X_train.shape)
    print(y_train.shape)
    num_epoch = 500

    model = LogisticRegression()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    # define metric
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
    test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')

    # Train the model
    for epoch in range(num_epoch):
        train_loss.reset_states()
        train_accuracy.reset_states()

        train_step(X_train, y_train)
        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            f'Accuracy: {train_accuracy.result() * 100}.'
        )

    # test the model
    print()
    print("Model Evaluation: ")
    test_step(X_test, y_test)
    print(
        f'Test Accuracy: {test_accuracy.result() * 100}.'
    )