import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


class Layer:
    """One layer of the network, it is output neurons and preceding weights"""

    def __init__(self, n_input: int, n_output: int, activation_type: str = "sigmoid"):
        self.weights = np.random.rand(
            n_output, n_input
        )  # matrix of weights. Rows corresponds to output neurons, columns to input neurons
        self.bias = np.random.rand(n_output).reshape((-1, 1))
        self.activation_type = activation_type
        self.n_input = n_input
        self.n_output = n_output

    def set_weights(self, W):
        self.weights = np.array(W).reshape(self.weights.shape)

    def set_bias(self, b):
        self.bias = np.array(b).reshape(self.bias.shape)

    def activation(self, x):
        if self.activation_type == "sigmoid":
            return 1 / (1 + np.exp(-x))

    def derivative_activation(self, x):
        if self.activation_type == "sigmoid":
            return np.exp(-x) / (1 + np.exp(-x)) ** 2

    def output(self, inputs: np.array):
        """Returns column vector of outputs calculated for given weights and activation function"""
        return self.activation(self.lin_comb(inputs))

    def lin_comb(self, inputs):
        """Returns weights(matrix) * inputs (column) + bias (column)"""
        inputs = inputs.reshape((-1, 1))  # column vector
        W = self.weights
        Wx = np.matmul(W, inputs).reshape((-1, 1))  # column vector
        return (Wx + self.bias).reshape((-1, 1))  # column vector

    def set_delta(self, delta):
        self.delta = np.array(delta).reshape((-1, 1))


class Network:
    def __init__(
        self,
        layers: np.array,
        activation_type="sigmoid",
        alpha=0.1,
        batch_size=10,
        n_epochs=10,
    ):
        self.alpha = alpha
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        layers_kwargs = {"activation_type": activation_type}
        try:
            if len(layers) < 2:
                raise ValueError("Network must have at least 2 layers")
        except TypeError:  # if len(layers) throws error (for example user specified an integer)
            raise TypeError("Layers must be a list of number of neurons in each layer")
        self.layers = [
            Layer(layers[i], layers[i + 1], **layers_kwargs)
            for i in range(len(layers) - 1)
        ]

    def train(self, X, Y):
        for _ in range(self.n_epochs):  # repeat on whole dataset n_epochs times
            X = np.array(X)
            Y = np.array(Y)
            if Y.ndim == 1:
                Y = Y.reshape(
                    (-1, 1)
                )  # if y is an one-dimensional vector - make it a column vector (matrix)
            if X.shape[0] != Y.shape[0]:
                raise ValueError("X and y have different row numbers")
            n_rows = X.shape[0]
            for i in range(n_rows):
                x = X[i, :].reshape((1, -1))  # row vector
                y = Y[i, :].reshape((1, -1))  # row vector
                pred = self.fit(x)
                # going back to front - setting deltas
                for n, layer in reversed(list(enumerate(self.layers))):
                    if n == len(self.layers) - 1:  # if this is output layer
                        # pairwise multiplication, not matrix multiplication
                        delta = (pred * (1 - pred)).reshape((-1, 1)) * (
                            self.fit(x) - y
                        ).reshape((-1, 1))
                        layer.set_delta(delta)

                    else:  # if this is a hidden layer
                        print(f"I am in hidden {n}")
                        delta =np.matmul(prev_weights.transpose(), delta).reshape(-1, 1)
                        layer.set_delta(delta)  # using delta and weights from n+1
                    prev_weights = layer.weights # we calculate delta in n layer using weights from n+1

                # going front to back - updating weights using deltas
                for layer in self.layers:
                    y = layer.output(x)
                    layer.set_weights(
                        layer.weights
                        - self.alpha * np.matmul(layer.delta, x.reshape((1, -1)))
                    )  #
                    layer.set_bias(layer.bias - self.alpha * layer.delta)
                    x = y  # output becomes input for next layer

    def fit(self, X):
        def fit_one(x):
            y = x
            for layer in self.layers:
                y = layer.output(
                    y
                )  # output from previous layer becomes input for next layer
            return y.reshape((-1,))

        return np.apply_along_axis(fit_one, axis=1, arr=X)


X, y = load_iris(return_X_y=True)
encoder = preprocessing.OneHotEncoder()
y = encoder.fit_transform(y.reshape((-1, 1))).todense()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)
N = Network([4, 5, 6, 3], alpha=0.4, n_epochs=40)
N.train(X_train, y_train)
pred = N.fit(X_test)
print(np.argmax(pred, axis=1).reshape((-1,)))
print(np.argmax(y_test, axis=1).reshape((-1,)))
print(
    np.mean(
        np.argmax(pred, axis=1).reshape((-1)) == np.argmax(y_test, axis=1).reshape((-1))
    )
)
