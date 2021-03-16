import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from network import Network

PATH = os.path.join("data", "classification")
datasets = ["data.simple", "data.three_gauss"]
sizes = [100, 500, 1000, 10000]
N = 100
fig, axs = plt.subplots(3, 4, figsize=(12, 8))
fig.tight_layout(pad=4.0)
for j, size in enumerate(sizes):
    data = pd.read_csv(os.path.join(PATH, f"{datasets[0]}.train.{size}.csv"))
    mesh = (
        np.mgrid[
            min(data.x) : max(data.x) : (max(data.x) - min(data.x)) / N,
            min(data.y) : max(data.y) : (max(data.y) - min(data.y)) / N,
        ]
        .reshape(2, -1)
        .T
    )
    cls = preprocessing.OneHotEncoder().fit_transform(data[["cls"]]).todense()
    MLP = Network([2, cls.shape[1]])
    MLP.train(data[["x", "y"]].to_numpy(), cls)
    pred_mesh = np.argmax(MLP.fit(mesh), axis=1)
    axs[0][j].scatter(mesh[:, 0], mesh[:, 1], s=0.5, c=pred_mesh)
    axs[0][j].scatter(data.x, data.y, c=data.cls, cmap="ocean", s=2)
    axs[0][j].set_title(f"Network: {[2, 1]}, size: {size}")
for j, size in enumerate(sizes):
    data = pd.read_csv(os.path.join(PATH, f"{datasets[1]}.train.{size}.csv"))
    mesh = (
        np.mgrid[
            min(data.x) : max(data.x) : (max(data.x) - min(data.x)) / N,
            min(data.y) : max(data.y) : (max(data.y) - min(data.y)) / N,
        ]
        .reshape(2, -1)
        .T
    )
    cls = preprocessing.OneHotEncoder().fit_transform(data[["cls"]]).todense()
    MLP = Network([2, cls.shape[1]])
    MLP.train(data[["x", "y"]].to_numpy(), cls)
    pred_mesh = np.argmax(MLP.fit(mesh), axis=1)
    axs[1][j].scatter(mesh[:, 0], mesh[:, 1], s=0.5, c=pred_mesh)
    axs[1][j].scatter(data.x, data.y, c=data.cls, cmap="ocean", s=2)
    axs[1][j].set_title(f"Network: {[2, 1]}, size: {size}")
for j, size in enumerate(sizes):
    data = pd.read_csv(os.path.join(PATH, f"{datasets[1]}.train.{size}.csv"))
    mesh = (
        np.mgrid[
            min(data.x) : max(data.x) : (max(data.x) - min(data.x)) / N,
            min(data.y) : max(data.y) : (max(data.y) - min(data.y)) / N,
        ]
        .reshape(2, -1)
        .T
    )
    cls = preprocessing.OneHotEncoder().fit_transform(data[["cls"]]).todense()
    MLP = Network([2, 4, cls.shape[1]])
    MLP.train(data[["x", "y"]].to_numpy(), cls)
    pred_mesh = np.argmax(MLP.fit(mesh), axis=1)
    axs[2][j].scatter(mesh[:, 0], mesh[:, 1], s=0.5, c=pred_mesh)
    axs[2][j].scatter(data.x, data.y, c=data.cls, cmap="ocean", s=2)
    axs[2][j].set_title(f"Network: {[2, 4, 1]}, size: {size}")
plt.show()
