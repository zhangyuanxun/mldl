import tensorflow as tf
import numpy as np
from model import FM
from data_loader import *
import os


def train_step(X, y):
    with tf.GradientTape() as tape:
        y_hat = model(X, training=True)
        loss = loss_fn(y, y_hat)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return y_hat, loss


def test_step(X, y):
    out = model(X, training=False)
    pred = tf.round(out)  # return 0 or 1
    return pred


if __name__ == "__main__":
    movielens_dataset = MovieLensDataset(train_rating_path=RATING_FILE_PATH_TRAIN,
                                         test_rating_path=RATING_FILE_PATH_TEST,
                                         user_path=USER_FILE_PATH,
                                         item_path=ITEM_FILE_PATH)

    X_train, X_test, y_train, y_test = movielens_dataset.load_data()
    field_dims = movielens_dataset.field_dims
    print("The size of train dataset is: {}, and the size of test dataset is: {}".format(X_train.shape[0],
                                                                                         X_test.shape[0]))
    embedding_dim = 10
    num_epochs = 50
    batch_size = 64

    # load dataset to the tensor
    X_train = tf.data.Dataset.from_tensor_slices(X_train)
    y_train = tf.data.Dataset.from_tensor_slices(y_train)

    train_dataset = tf.data.Dataset.zip((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=100)
    train_dataset = train_dataset.batch(batch_size)

    X_test = tf.data.Dataset.from_tensor_slices(X_test)
    y_test = tf.data.Dataset.from_tensor_slices(y_test)

    test_dataset = tf.data.Dataset.zip((X_test, y_test))
    test_dataset = test_dataset.batch(batch_size)

    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'

    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{step}")

    model = FM(field_dims=field_dims, embedding_dim=embedding_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.losses.BinaryCrossentropy(from_logits=False)

    train_loss_results = []
    train_accuracy_results = []

    # train the model
    best_test_acc = 0
    for epoch in range(num_epochs):
        epoch_loss = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.BinaryAccuracy()

        for nb, (X, y) in enumerate(train_dataset):
            y_hat, loss = train_step(X, y)
            # update metrics
            epoch_loss.update_state(loss)
            epoch_accuracy.update_state(y, y_hat)

        # End epoch
        train_loss_results.append(epoch_loss.result())
        train_accuracy_results.append(epoch_accuracy.result())

        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch + 1,
                                                                    epoch_loss.result(),
                                                                    epoch_accuracy.result()))

        # evaluate the model and save the best model
        test_accuracy_metric = tf.keras.metrics.Accuracy()
        for nb, (X, y) in enumerate(test_dataset):
            pred = test_step(X, y)
            test_accuracy_metric(pred, y)

        cur_test_acc = test_accuracy_metric.result()
        print("Test set accuracy: {:.3%}".format(cur_test_acc))
        if cur_test_acc > best_test_acc:
            model.save_weights(checkpoint_prefix.format(step=epoch))
            best_test_acc = cur_test_acc



