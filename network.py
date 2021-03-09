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
            (n_output, n_input)
        )  # matrix of weights. Rows corresponds to output neurons, columns to input neurons
        self.bias = np.random.normal(0, init_sigma, n_output).reshape((-1, 1))
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
        if self.activation_type == "tanh":
            return np.tanh(x)
        if self.activation_type == "relu":
            return np.maximum(0.0, x)

    def fit(self, inputs: np.array):
        """Returns output (sigma(Wx+b))"""
        return self.activation(self.lin_comb(np.array(inputs)))

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
        init_sigma=1,
        alpha=0.1,
        n_epochs=10,
        cost_fun="quadratic",
    ):
        self.cost_fun = cost_fun
        self.alpha = alpha
        self.n_epochs = n_epochs
        layers_kwargs = {"activation_type": activation_type, "init_sigma": init_sigma}
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
                # going from the back to the front - setting delta
                for n, layer in reversed(list(enumerate(self.layers))):
                    if n == len(self.layers) - 1:  # if this is output layer
                        # pairwise multiplication, not matrix multiplication
                        deriv = self.activation_derivative(layer, pred)

                        if self.cost_fun == "quadratic":
                            cost_deriv = pred - y
                        elif self.cost_fun == "cross-entropy":
                            # only with sigmoid activation function
                            # not sure if it's ok
                            cost_deriv = pred - y
                            deriv = np.array([1 for i in range(len(deriv))])
                        elif self.cost_fun == "hellinger":
                            # only with positive activation functions
                            cost_deriv = (np.sqrt(pred) - np.sqrt(y)) / (
                                np.sqrt(2) * np.sqrt(pred)
                            )
                        else:
                            raise ValueError("No such cost function")

                        delta = deriv.reshape((-1, 1)) * cost_deriv.reshape((-1, 1))

                        layer.set_delta(delta)

                    else:  # if this is a hidden layer
                        # using delta and weights from n+1
                        delta = np.matmul(prev_weights.transpose(), delta).reshape(
                            -1, 1
                        )
                        layer.set_delta(delta)
                    prev_weights = (
                        layer.weights
                    )  # we calculate delta in n layer using weights from n+1

                # going front to back - updating weights using deltas
                for layer in self.layers:
                    y = layer.fit(x)
                    layer.set_weights(
                        layer.weights
                        - self.alpha * np.matmul(layer.delta, x.reshape((1, -1)))
                    )  #
                    layer.set_bias(layer.bias - self.alpha * layer.delta)
                    x = y  # output becomes input for next layer

    def activation_derivative(self, layer, pred):
        if layer.activation_type == "sigmoid":
            deriv = pred * (1 - pred)
        elif layer.activation_type == "tanh":
            deriv = 1 - pred ** 2
        elif layer.activation_type == "relu":
            deriv = pred > 0
        else:
            raise ValueError("No such activation function")
        return deriv

    def fit(self, X):
        def fit_one(x):
            y = x
            for layer in self.layers:
                y = layer.fit(
                    y
                )  # output from previous layer becomes input for next layer
            return y.reshape((-1,))

        return np.apply_along_axis(fit_one, axis=1, arr=X)
