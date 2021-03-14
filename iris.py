from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from network import Network

np.random.seed(10)

np.random.seed(10)

print("----------------IRIS--------------------")
X, y = load_iris(return_X_y=True)
encoder = preprocessing.OneHotEncoder()
y = encoder.fit_transform(y.reshape((-1, 1))).todense()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
N = Network(
    [4, 3], learning_rate=0.01, n_epochs=400, cost_fun="cross-entropy"
)  # best without hidden layers - iris dataset is too small
for _ in range(10000):
    o = N.fit(X_train)
    error = np.array(o - y_train)
    N.layers[-1].set_delta(error * N.activation_derivative(o))
    for i in range(len(N.layers) - 2, -1, -1):
        if N.verbose:
            print("Updating deltas", i)
        error = np.dot(N.layers[i + 1].delta, N.layers[i + 1].weights.T)
        if N.verbose:
            print("Error", i)
            pprint(error)
        N.layers[i].set_delta(error * N.activation_derivative(N.layers[i].outputs))
    N.layers[0].update_weights(
        np.dot(X_train.T, N.layers[0].delta), N.learning_rate, N.momentum_rate
    )
    for i in range(1, len(N.layers)):
        N.layers[i].update_weights(
            np.dot(N.layers[i - 1].outputs.T, N.layers[i].delta),
            N.learning_rate,
            N.momentum_rate,
        )
pred = N.fit(X_test)
print(np.argmax(pred, axis=1).reshape((-1,)))
print(np.argmax(np.array(y_test), axis=1).reshape((-1,)))
print(
    np.mean(
        np.argmax(pred, axis=1).reshape((-1)) == np.argmax(y_test, axis=1).reshape((-1))
    )
)
