import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


class Layer:
    """One layer of the network, it is output neurons and preceding weights"""

    def __init__(self, n_input: int, n_output: int, activation: str = "sigmoid"):
        self.weights = np.random.rand(
            n_output, n_input
        )  # matrix of weights. Rows corresponds to output neurons, columns to input neurons
        self.bias = np.random.rand(n_output).reshape((-1, 1))
        self.activation_type = activation

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

    def delta(self, inputs: np.array, expectation: np.array):
        """Returns column vector of delta values. Each delta value corresponds to one output neuron"""
        expectation = np.array(expectation).reshape((-1, 1))  # column vector
        inputs = np.array(inputs).reshape((-1, 1))  # column vector
        # pairwise multiplication, not matrix multiplication
        return self.derivative_activation(self.lin_comb(inputs)).reshape((-1, 1)) * (
            self.output(inputs) - expectation
        ).reshape((-1, 1))


X, y = load_iris(return_X_y=True)
y = preprocessing.OneHotEncoder().fit_transform(y.reshape((-1, 1))).todense()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)
L = Layer(4, 3)
for i in range(X_train.shape[0]):
    delta = L.delta(X_train[i, :], y_train[i, :])
    L.set_weights(L.weights - 0.4 * np.matmul(delta, X_train[i, :].reshape((1, -1))))
    L.set_bias(L.bias - 0.4 * delta)
