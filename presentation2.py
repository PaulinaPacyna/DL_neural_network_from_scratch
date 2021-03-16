import os

import matplotlib.pyplot as plt
import pandas as pd

from network import Network
from sklearn.metrics import r2_score

PATH = os.path.join("data_presentation", "Regression")
datasets = ["data.linear", "data.multimodal", "data.square"]

sizes = [100, 500, 1000, 10000]
N = 100
fig, axs = plt.subplots(3, len(sizes), figsize=(18, 12))
fig.tight_layout(pad=5.0)
for j, size in enumerate(sizes):
    train = pd.read_csv(os.path.join(PATH, f"{datasets[0]}.train.{size}.csv"))
    test = pd.read_csv(os.path.join(PATH, f"{datasets[0]}.test.{size}.csv"))
    MLP = Network(
        [1, 2, 2, 1],
        regression=True,
        n_epochs=1000,
        batch_size=size // 100,
        activation_type="sigmoid",
        learning_rate=0.01,
        momentum_rate=0.01,
        print_progress=True,
    )
    MLP.train(train[["x"]].to_numpy(), train[["y"]].to_numpy())
    prediction = MLP.fit(test[["x"]].to_numpy(), predict=True)

    p = axs[0][j].scatter(test[["x"]], test[["y"]], c="red", facecolors="None")
    r = axs[0][j].scatter(train[["x"]], train[["y"]], c="green")
    q = axs[0][j].scatter(test[["x"]], prediction, c="blue")
    axs[0][j].legend([p, r, q], ["Test data", "Training data", "Prediction data"])
    r2 = round(r2_score(test[["y"]], prediction), 4)
    axs[0][j].set_title(
        f"R-squared score: {r2}, \nfor {size} obs \nand network {[1,2,2,1]}"
    )

for j, size in enumerate(sizes):
    train = pd.read_csv(os.path.join(PATH, f"{datasets[1]}.train.{size}.csv"))
    test = pd.read_csv(os.path.join(PATH, f"{datasets[1]}.test.{size}.csv"))
    MLP = Network(
        [1, 4, 8, 4, 1],
        regression=True,
        n_epochs=1000,
        batch_size=10,
        activation_type="sigmoid",
        learning_rate=0.01,
        momentum_rate=0.01,
        print_progress=True,
    )
    MLP.train(train[["x"]].to_numpy(), train[["y"]].to_numpy())
    prediction = MLP.fit(test[["x"]].to_numpy(), predict=True)

    p = axs[1][j].scatter(test[["x"]], test[["y"]], c="red", facecolors="None")
    r = axs[1][j].scatter(train[["x"]], train[["y"]], c="green")
    q = axs[1][j].scatter(test[["x"]], prediction, c="blue")
    axs[1][j].legend([p, r, q], ["Test data", "Training data", "Prediction data"])
    r2 = round(r2_score(test[["y"]], prediction), 4)
    axs[1][j].set_title(
        f"R-squared score: {r2}, \nfor {size} obs \nand network {[1,4,8,4,1]}"
    )

for j, size in enumerate(sizes):
    train = pd.read_csv(os.path.join(PATH, f"{datasets[2]}.train.{size}.csv"))
    test = pd.read_csv(os.path.join(PATH, f"{datasets[2]}.test.{size}.csv"))
    MLP = Network(
        [1, 3, 3, 1],
        regression=True,
        n_epochs=1000,
        batch_size=size // 100,
        activation_type="sigmoid",
        learning_rate=0.01,
        momentum_rate=0.01,
        print_progress=True,
    )
    MLP.train(train[["x"]].to_numpy(), train[["y"]].to_numpy())
    prediction = MLP.fit(test[["x"]].to_numpy(), predict=True)

    p = axs[2][j].scatter(test[["x"]], test[["y"]], c="red", facecolors="None")
    r = axs[2][j].scatter(train[["x"]], train[["y"]], c="green")
    q = axs[2][j].scatter(test[["x"]], prediction, c="blue")
    axs[2][j].legend([p, r, q], ["Test data", "Training data", "Prediction data"])
    r2 = round(r2_score(test[["y"]], prediction), 4)
    axs[2][j].set_title(
        f"R-squared score: {r2}, \nfor {size} obs \nand network {[1,3,3,1]}"
    )
plt.show()
