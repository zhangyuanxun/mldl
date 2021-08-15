import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class LinearRegression(tf.keras.Model):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.W = tf.keras.layers.Dense(1, use_bias=True, kernel_initializer='random_uniform')

    def call(self, X):
        return self.W(X)


def train_step(X, Y):
    with tf.GradientTape() as tape:
        Y_hat = model(X)
        loss = loss_fn(Y, Y_hat)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(X, Y)


if __name__ == '__main__':
    # generate random dataset
    num_points = 1000
    points = []
    W, b = 0.1, 0.3
    for i in range(num_points):
        x = np.random.normal(0.0, 0.55)
        y = x * W + b + np.random.normal(0.0, 0.03)
        points.append([x, y])

    # generate x, y dataset
    x_data = [p[0] for p in points]
    y_data = [p[1] for p in points]

    x_data = tf.expand_dims(tf.convert_to_tensor(x_data), axis=1)
    y_data = tf.convert_to_tensor(y_data)

    # plot this line
    plt.scatter(x_data, y_data, c='r')
    plt.show()
    num_epoch = 50

    # define model
    model = LinearRegression()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.3)
    loss_fn = tf.keras.losses.MeanSquaredError()

    # define metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.MeanSquaredError(name='train_accuracy')

    for epoch in range(num_epoch):
        train_step(x_data, y_data)

    print("Input Weight:")
    print("W = %.4f, b = %.4f" % (W, b))
    
    print("Final Weight:")
    W = model.layers[0].get_weights()[0]
    b = model.layers[0].get_weights()[1]
    print("W = %.4f, b = %.4f" % (W, b))

    # plot the fitted line
    plt.scatter(x_data, y_data, c='r')
    plt.plot(x_data, W * x_data + b, c='b')
    plt.show()



