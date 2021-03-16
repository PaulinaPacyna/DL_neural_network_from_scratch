import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from network import Network

np.random.seed(10)

print("----------------IRIS--------------------")
X, y = load_iris(return_X_y=True)
encoder = preprocessing.OneHotEncoder()
y = encoder.fit_transform(y.reshape((-1, 1))).todense()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
N = Network(
    [4, 3], learning_rate=0.01, n_epochs=4000, cost_fun="cross-entropy"
)  # best without hidden layers - iris dataset is too small
N.train(X_train, y_train)
pred = N.fit(X_test)
print("Prediction: ", np.argmax(pred, axis=1).reshape((-1,)))
print("True :", np.argmax(np.array(y_test), axis=1).reshape((-1,)))
print(
    "Accuracy :",
    np.mean(
        np.argmax(pred, axis=1).reshape((-1)) == np.argmax(y_test, axis=1).reshape((-1))
    ),
)


np.random.seed(10)
