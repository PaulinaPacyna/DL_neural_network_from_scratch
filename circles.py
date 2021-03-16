import numpy as np
import matplotlib.pyplot as plt
from network import Network


def add_circle(xs, ys, a, b, r, val, n, r2=0):
    r = np.random.uniform(r2, r, n)
    phi = np.random.uniform(0, 360, n)
    x = r * np.cos(phi) + a
    y = r * np.sin(phi) + b
    xy = np.array([x, y]).T
    return np.concatenate([xs, xy], axis=0), np.concatenate(
        [ys, val * np.ones((xy.shape[0], 1))],
        axis=0,
    )


X = np.zeros((0, 2))
y = np.zeros((0, 1))
X, y = add_circle(X, y, 0, 0, 5, 0, 400, r2=2.5)
X, y = add_circle(X, y, 0, 0, 2, 1, 400)
X, y = add_circle(X, y, -4.5, -4.5, 1.5, 1, 100)
X, y = add_circle(X, y, -4.5, 4.5, 1.5, 1, 100)
X, y = add_circle(X, y, 4.5, -4.5, 1.5, 1, 100)
X, y = add_circle(X, y, 4.5, 4.5, 1.5, 1, 100)
NN = Network(
    [X.shape[1], 4, 4, 3, 1],
    learning_rate=0.01,
    momentum_rate=0,
    n_epochs=10000,
    batch_size=200,
    print_progress=True,
)
NN.train(X, y)
mesh = np.mgrid[-6:6:0.1, -6:6:0.1].reshape(2, -1).T
plt.scatter(mesh[:, 0], mesh[:, 1], c=NN.fit(mesh), alpha=0.3, cmap="coolwarm")
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title(f"Network architecture: {[X.shape[1], 4, 4, 3, 1]}")
