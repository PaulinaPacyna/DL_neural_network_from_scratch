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
    [4, 3], learning_rate=0.25, n_epochs=5, batch_size=10
)  # best without hidden layers - iris dataset is too small
N.train_batches(X_train, y_train)
pred = N.fit(X_test)
print(np.argmax(pred, axis=1).reshape((-1,)))
print(np.argmax(np.array(y_test), axis=1).reshape((-1,)))
print(
    np.mean(
        np.argmax(pred, axis=1).reshape((-1)) == np.argmax(y_test, axis=1).reshape((-1))
    )
)


np.random.seed(10)

print("----------------XOR--------------------")
# sigmoid do xora to kiepski pomysl - w teorii jest zawsze zbiezne, a w praktyce to na 30 razy raz sie zbieglo
# internet poleca inne activation function (tanh, relu)  - do zrobienia
data = np.array([[1, 1, 0], [1, 0, 1], [0, 0, 0], [0, 1, 1]])
X = data[:, :2]
y = data[:, 2]
N = Network(
    [2, 2, 1],
    learning_rate=0.9,
    activation_type="sigmoid",
    n_epochs=1000,
    init_sigma=4,
    cost_fun="hellinger",
)
N.train(X, y)
pred = N.fit(np.array([[1, 1], [1, 0], [0, 0], [0, 1]]))
xy = np.mgrid[-1:1.1:0.05, -1:1.1:0.05].reshape(2, -1).T
plt.scatter(xy[:, 0], xy[:, 1], c=np.round(N.fit(xy)), s=1)
plt.show()
