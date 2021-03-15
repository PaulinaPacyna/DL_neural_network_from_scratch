from copy import copy

import numpy as np


class Layer:
    """One layer of the network, it is output neurons and preceding weights"""

    def __init__(
        self,
        n_input: int,
        n_output: int,
        activation_type: str = "sigmoid",
        init_sigma=1,
    ):
        self.weights = np.random.normal(0, init_sigma, n_output * n_input).reshape(
            (n_input, n_output)
        )
        self.bias = np.random.normal(0, init_sigma, n_output).reshape((1, -1))
        self.activation_type = activation_type
        self.n_input = n_input
        self.n_output = n_output
        self.momentum = np.zeros(self.weights.shape)
        self.momentum_bias = np.zeros(self.bias.shape)
        self.outputs = np.zeros(n_output)

    def update_weights(self, W, learning_rate, momentum_rate):
        change = (
            learning_rate * np.array(W).reshape(self.weights.shape)
            + momentum_rate * self.momentum
        )
        self.weights = self.weights - change
        self.momentum = change

    def update_bias(self, b, learning_rate, momentum_rate):
        change = (
            learning_rate * np.array(b).reshape(self.bias.shape)
            + momentum_rate * self.momentum_bias
        )
        self.bias = self.bias - change
        self.momentum_bias = change

    def activation(self, x):
        if self.activation_type == "sigmoid":
            self.outputs = 1 / (1 + np.exp(-x))
        if self.activation_type == "tanh":
            self.outputs = np.tanh(x)
        if self.activation_type == "relu":
            self.outputs = np.maximum(0.0, x)
        if self.activation_type == "linear":
            self.outputs = np.array(x)
        return self.outputs

    def fit(self, inputs: np.array):
        """Returns output (sigma(Wx+b))"""
        return self.activation(self.lin_comb(np.array(inputs)))

    def lin_comb(self, inputs):
        """Returns weights(matrix) * inputs (column) + bias (column)"""
        Wx = np.matmul(inputs, self.weights)
        return Wx + self.bias

    def set_delta(self, delta):
        self.delta = np.array(delta)


class Network:
    def __init__(
        self,
        layers: np.array,
        activation_type="sigmoid",
        init_sigma=1,
        learning_rate=0.1,
        momentum_rate=0.1,
        n_epochs=100,
        cost_fun="quadratic",
        batch_size=10,
        print_progress=False,
        regression=False,
    ):
        self.cost_fun = cost_fun
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.print_progress = print_progress
        self.regression = regression
        self.x_min, self.x_maxmin = 0, 1
        self.y_min, self.y_maxmin = 0, 1
        layers_kwargs = {"activation_type": activation_type, "init_sigma": init_sigma}
        try:
            if (
                len(layers) < 2
            ):  # if len(layers) throws error (for example user specified an integer)
                raise ValueError("Network must have at least 2 layers")
        except TypeError:
            raise TypeError("Layers must be a list of number of neurons in each layer")
        if regression:
            if layers[-1] != 1:
                raise ValueError(
                    "In regression problem, output layer consists of 1 neuron"
                )
            self.layers = [
                Layer(layers[i], layers[i + 1], **layers_kwargs)
                for i in range(len(layers) - 2)
            ]
            self.layers.append(Layer(layers[-2], layers[-1], "linear", init_sigma))
        else:
            self.layers = [
                Layer(layers[i], layers[i + 1], **layers_kwargs)
                for i in range(len(layers) - 1)
            ]

    def train(self, X, Y):
        X = np.array(X)
        Y = np.array(Y)
        if self.regression:
            X = self.scale_x(X)
            Y = self.scale_y(Y)
        if Y.ndim == 1:
            Y = Y.reshape(
                (-1, 1)
            )  # if y is an one-dimensional vector - make it a column vector (matrix)
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and y have different row numbers")
        n_row = X.shape[0]
        shuffle = np.arange(Y.shape[0])
        np.random.shuffle(shuffle)
        X = X[shuffle, :]
        Y = Y[shuffle, :]
        for e in range(self.n_epochs):
            batch_division = np.arange(self.batch_size, n_row, self.batch_size)
            x_batches = np.split(X, batch_division)
            y_batches = np.split(Y, batch_division)
            for n_batch in range(len(batch_division)):
                x = x_batches[n_batch]
                y = y_batches[n_batch]
                prediction = self.fit(x)
                # computing delta in last layer using standard chain rule
                self.layers[-1].set_delta(
                    self.delta(prediction, y)
                )
                # going back to front starting from last hidden layer
                for n_layer in range(len(self.layers) - 2, -1, -1):
                    error = np.matmul(
                        self.layers[n_layer + 1].delta,
                        self.layers[n_layer + 1].weights.T,
                    )
                    self.layers[n_layer].set_delta(
                        error
                        * self.activation_derivative(
                            self.layers[n_layer], self.layers[n_layer].outputs
                        )
                    )
                # going front to back - updating weights using deltas that we just computed
                self.layers[0].update_weights(
                    np.matmul(x.T, self.layers[0].delta),
                    self.learning_rate,
                    self.momentum_rate,
                )
                self.layers[0].update_bias(
                    self.layers[0].delta.sum(axis=0)[None, :],
                    self.learning_rate,
                    self.momentum_rate,
                )
                for n_layer in range(1, len(self.layers)):
                    self.layers[n_layer].update_weights(
                        np.matmul(
                            self.layers[n_layer - 1].outputs.T,
                            self.layers[n_layer].delta,
                        ),
                        self.learning_rate,
                        self.momentum_rate,
                    )
                    self.layers[n_layer].update_bias(
                        self.layers[n_layer].delta.sum(axis=0)[None, :],
                        self.learning_rate,
                        self.momentum_rate,
                    )
            if self.print_progress:
                if e % 1000 == 0:
                    print(f"Epoch: {e}/{self.n_epochs}")

    def fit(self, X, predict=False):
        y = (X-self.x_min)/self.x_maxmin if predict else copy(X)
        for layer in self.layers:
            y = layer.fit(y)
        if predict:
            return self.rescale_y(y)
        return y

    def scale_x(self, X):
        self.x_min = min(X)
        self.x_maxmin = max(X) - min(X)
        return (X - self.x_min) / self.x_maxmin

    def scale_y(self, Y):
        self.y_min = min(Y)
        self.y_maxmin = max(Y) - min(Y)
        return (Y - self.y_min) / self.y_maxmin

    def rescale_x(self, X):
        return X * self.x_maxmin + self.x_min

    def rescale_y(self, Y):
        return Y * self.y_maxmin + self.y_min


    def delta(self, a, y):
        if self.cost_fun == "quadratic":
            return np.array(a-y) * self.activation_derivative(self.layers[-1], a)
        elif self.cost_fun == "cross-entropy":
            return np.array(a-y)
        elif self.cost_fun == "hellinger":
            if self.layers[-1].activation_type != "sigmoid":
                raise ValueError("Hellinger cost function works only with (0,1) values of activation!")
            return ((np.sqrt(a) - np.sqrt(y))/(np.sqrt(2)*np.sqrt(a))) * self.activation_derivative(self.layers[-1], a)
        else:
            raise ValueError("No such cost function!")
    @staticmethod
    def activation_derivative(layer, pred):
        if layer.activation_type == "sigmoid":
            deriv = pred * (1 - pred)
        elif layer.activation_type == "tanh":
            deriv = 1 - pred ** 2
        elif layer.activation_type == "relu":
            deriv = (pred > 0) + 0  # this changes to int
        elif layer.activation_type == "linear":
            deriv = np.ones(pred.shape)
        else:
            raise ValueError("No such activation function")
        return deriv
