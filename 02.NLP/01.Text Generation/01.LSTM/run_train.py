from data_loader import Vocabulary, batch_generator
from model import LSTMModel
import tensorflow as tf
import time
import os


def train_step(X, Y):
    with tf.GradientTape() as tape:
        Y_hat = model(X, training=True)
        loss = loss_fn(Y, Y_hat)
    gradients = tape.gradient(loss, model.trainable_variables)
    gradients, _ = tf.clip_by_global_norm(gradients, grad_clip)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)


if __name__ == "__main__":
    # load files
    datafile = "../../../00.Datasets/01.NLP/01.misc/jay_lyric.txt"
    with open(datafile, 'r', encoding='utf-8') as f:
        train_data = f.read()

    # build Vocabulary
    vocabulary = Vocabulary()
    vocabulary.build_vocab(train_data)
    vocabulary.save('vocab.pkl')

    # define hyper parameters
    batch_size = 32
    seq_len = 20
    num_epoch = 10
    grad_clip = 5.0
    max_step = 20000

    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'

    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{step}")

    input_ids = vocabulary.encode(train_data)
    batch_data = batch_generator(input_ids, batch_size, seq_len)

    model = LSTMModel(vocabulary.vocab_size, batch_size=batch_size, num_steps=seq_len,
                      lstm_size=128, num_layers=2, sampling=False, drop_out=0.5,
                      use_embedding=False, embedding_size=128)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

    # define metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    step = 0
    for nb, (X, y) in enumerate(batch_data):
        start = time.time()
        train_loss.reset_states()
        step += 1
        train_step(X, y)
        if nb % 10 == 0:
            print(f"Step {step} Batch {nb} Loss {train_loss.result().numpy():.4f}")

        # saving (checkpoint) the model every 5 epochs
        if (step + 1) % 1000 == 0:
            model.save_weights(checkpoint_prefix.format(step=step + 1))

        if step >= max_step:
            break

    model.save_weights(checkpoint_prefix.format(step=step))





