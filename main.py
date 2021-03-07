import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from network import Network, Layer

print("----------------IRIS--------------------")
X, y = load_iris(return_X_y=True)
encoder = preprocessing.OneHotEncoder()
y = encoder.fit_transform(y.reshape((-1, 1))).todense()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
N = Network(
    [4, 3], alpha=0.4, n_epochs=400
)  # best without hidden layers - iris dataset is too small
N.train(X_train, y_train)
pred = N.fit(X_test)
print(np.argmax(pred, axis=1).reshape((-1,)))
print(np.argmax(np.array(y_test), axis=1).reshape((-1,)))
print(
    np.mean(
        np.argmax(pred, axis=1).reshape((-1)) == np.argmax(y_test, axis=1).reshape((-1))
    )
)
print("----------------XOR--------------------")
# sigmoid do xora to kiepski pomysl - w teorii jest zawsze zbiezne, a w praktyce to na 30 razy raz sie zbieglo
# internet poleca inne activation function (tanh, relu)  - do zrobienia
data = np.array([[1, 1, 0], [1, 0, 1], [0, 0, 0], [0, 1, 1]])
X = data[:, :2]
y = data[:, 2]
N = Network([2, 2, 1], alpha=0.9, n_epochs=10000, init_sigma=7)
N.train(X, y)
pred = N.fit(np.array([[1, 1], [1, 0], [0, 0], [0, 1]]))
xy = np.mgrid[-1:3.1:0.05, -1:3.1:0.05].reshape(2, -1).T
plt.scatter(xy[:, 0], xy[:, 1], c=np.round(N.fit(xy)), s=1)
plt.show()
