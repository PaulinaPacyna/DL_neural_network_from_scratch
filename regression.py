import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
from network import Network

PATH = os.path.join("data", "regression")
datasets = ["data.activation", "data.cube"]

sizes = [100, 500, 1000, 10000]
N = 100
fig, axs = plt.subplots(2, len(sizes), figsize=(12, 8))
for j, size in enumerate(sizes):
    train = pd.read_csv(os.path.join(PATH, f"{datasets[0]}.train.{size}.csv"))
    test = pd.read_csv(os.path.join(PATH, f"{datasets[0]}.test.{size}.csv"))
    MLP = Network(
        [1, 2, 2, 1],
        regression=True,
        n_epochs=10000,
        batch_size=size // 100,
        activation_type="sigmoid",
        learning_rate=0.01,
        momentum_rate=0.01,
        print_progress=True,
    )
    MLP.train(train[["x"]].to_numpy(), train[["y"]].to_numpy())
    prediction = MLP.fit(test[["x"]].to_numpy(), predict=True)

    p = axs[0][j].scatter(
        test[["x"]],
        test[["y"]],
        c="red",
    )
    r = axs[0][j].scatter(train[["x"]], train[["y"]], c="green")
    q = axs[0][j].scatter(test[["x"]], prediction, c="blue")
    axs[0][j].legend([p, r, q], ["Test data", "Training data", "Prediction data"])
    score = r2_score(test[["y"]], prediction)
    axs[0][j].set_title(
        f"Network architecture: {[1, 2, 2, 1]}, size: {size } r2 score: {score}",
    )

for j, size in enumerate(sizes):
    train = pd.read_csv(os.path.join(PATH, f"{datasets[1]}.train.{size}.csv"))
    test = pd.read_csv(os.path.join(PATH, f"{datasets[1]}.test.{size}.csv"))
    MLP = Network(
        [1, 3, 3, 1],
        regression=True,
        n_epochs=10000,
        batch_size=size // 100,
        activation_type="sigmoid",
        learning_rate=0.01,
        momentum_rate=0.01,
        print_progress=True,
    )
    MLP.train(train[["x"]].to_numpy(), train[["y"]].to_numpy())
    prediction = MLP.fit(test[["x"]].to_numpy(), predict=True)

    p = axs[1][j].scatter(test[["x"]], test[["y"]], c="red")
    r = axs[1][j].scatter(train[["x"]], train[["y"]], c="green")
    q = axs[1][j].scatter(test[["x"]], prediction, c="blue")
    axs[1][j].legend([p, r, q], ["Test data", "Training data", "Prediction data"])
    score = r2_score(test[["y"]], prediction)
    axs[1][j].set_title(
        f"Network architecture: {[1, 3, 3, 1]}, size: { size} r2 score:{score}"
    )


plt.show()
