import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing

from network import Network

PATH = os.path.join("data_presentation", "Classification")
datasets = ["data.circles", "data.XOR", "data.noisyXOR"]
sizes = [100, 500, 1000, 10000]
N = 100
fig, axs = plt.subplots(3, 4, figsize=(12, 8))
for j, size in enumerate(sizes):
    data = pd.read_csv(os.path.join(PATH, f"{datasets[0]}.train.{size}.csv"))
    data_test = pd.read_csv(os.path.join(PATH, f"{datasets[0]}.test.{size}.csv"))
    mesh = (
        np.mgrid[
            min(data.x): max(data.x): (max(data.x) - min(data.x)) / N,
            min(data.y): max(data.y): (max(data.y) - min(data.y)) / N,
        ]
        .reshape(2, -1)
        .T
    )
    cls = preprocessing.OneHotEncoder().fit_transform(data[["cls"]]).todense()
    MLP = Network([2, 4, 8, cls.shape[1]], n_epochs=100, batch_size=10, activation_type="sigmoid")
    MLP.train(data[["x", "y"]].to_numpy(), cls)
    pred_mesh = np.argmax(MLP.fit(mesh), axis=1)
    cls_test = preprocessing.OneHotEncoder().fit_transform(data_test[["cls"]]).todense()
    y_pred = np.argmax(MLP.fit(data_test[["x", "y"]].to_numpy()), axis=1)
    axs[0][j].scatter(mesh[:, 0], mesh[:, 1], s=0.5, c=pred_mesh)
    axs[0][j].scatter(data.x, data.y, c=data.cls, cmap="ocean", s=2)
    print(f"Accuracy of {datasets[0]} with {size} obs: ", sum((y_pred+1) == data_test["cls"])/len(y_pred))
for j, size in enumerate(sizes):
    data = pd.read_csv(os.path.join(PATH, f"{datasets[1]}.train.{size}.csv"))
    data_test = pd.read_csv(os.path.join(PATH, f"{datasets[1]}.test.{size}.csv"))
    mesh = (
        np.mgrid[
        min(data.x): max(data.x): (max(data.x) - min(data.x)) / N,
        min(data.y): max(data.y): (max(data.y) - min(data.y)) / N,
        ]
            .reshape(2, -1)
            .T
    )
    cls = preprocessing.OneHotEncoder().fit_transform(data[["cls"]]).todense()
    MLP = Network([2, 8, cls.shape[1]], n_epochs=100, batch_size=10, activation_type="relu")
    MLP.train(data[["x", "y"]].to_numpy(), cls)
    pred_mesh = np.argmax(MLP.fit(mesh), axis=1)
    y_pred = np.argmax(MLP.fit(data_test[["x", "y"]].to_numpy()), axis=1)
    axs[1][j].scatter(mesh[:, 0], mesh[:, 1], s=0.5, c=pred_mesh)
    axs[1][j].scatter(data.x, data.y, c=data.cls, cmap="ocean", s=2)
    print(f"Accuracy of {datasets[1]} with {size} obs: ", sum((y_pred+1) == data_test["cls"])/len(y_pred))

for j, size in enumerate(sizes):
    data = pd.read_csv(os.path.join(PATH, f"{datasets[2]}.train.{size}.csv"))
    data_test = pd.read_csv(os.path.join(PATH, f"{datasets[2]}.test.{size}.csv"))
    mesh = (
        np.mgrid[
        min(data.x): max(data.x): (max(data.x) - min(data.x)) / N,
        min(data.y): max(data.y): (max(data.y) - min(data.y)) / N,
        ]
            .reshape(2, -1)
            .T
    )
    cls = preprocessing.OneHotEncoder().fit_transform(data[["cls"]]).todense()
    MLP = Network([2, 8, cls.shape[1]], n_epochs=100, batch_size=10, activation_type="relu")
    MLP.train(data[["x", "y"]].to_numpy(), cls)
    pred_mesh = np.argmax(MLP.fit(mesh), axis=1)
    y_pred = np.argmax(MLP.fit(data_test[["x", "y"]].to_numpy()), axis=1)
    axs[2][j].scatter(mesh[:, 0], mesh[:, 1], s=0.5, c=pred_mesh)
    axs[2][j].scatter(data.x, data.y, c=data.cls, cmap="ocean", s=2)
    print(f"Accuracy of {datasets[2]} with {size} obs: ", sum((y_pred + 1) == data_test["cls"]) / len(y_pred))
plt.show()