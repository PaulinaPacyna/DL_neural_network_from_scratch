import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Layer:
    """One layer of the network, it is output neurons and preceding weights"""

    def __init__(self, n_input: int, n_output: int, activation: str = "sigmoid"):
        self.weights = np.random.rand(
            n_output, n_input
        )  # matrix of weights. Rows corresponds to output neurons, columns to input neurons
        self.bias = np.random.rand(n_output).reshape((-1, 1))
        self.activation_type = activation

    def set_weights(self, W):
        self.weights = W.reshape(self.weights.shape)

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
        expectation = expectation.reshape((-1, 1))  # column vector
        inputs = inputs.reshape((-1, 1))  # column vector
        # pairwise multiplication, not matrix multiplication
        return self.derivative_activation(self.lin_comb(inputs)) * (
            self.output(inputs) - expectation
        )


x = np.array([2, 3, 4])
L = Layer(3, 2)
y = L.output(x)
delta = L.delta(x, np.array([0.3, 0.7]))
