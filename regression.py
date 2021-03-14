import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import r2_score

from network import Network

PATH = os.path.join("data", "regression")
datasets = ["data.activation", "data.cube"]

sizes = [100, 500, 1000, 10000]
N = 500

np.random.seed(15)
fig, axs = plt.subplots(1, 4, figsize=(16, 10))
for j, size in enumerate(sizes):
    training_data = pd.read_csv(os.path.join(PATH, f"{datasets[0]}.train.{size}.csv"))
    test_data = pd.read_csv(os.path.join(PATH, f"{datasets[0]}.test.{size}.csv"))
    MLP = Network([1, 64, 1], regression=True, n_epochs=20, batch_size=16, activation_type="sigmoid")
    MLP.train(training_data[["x"]].to_numpy(), training_data[["y"]].to_numpy())
    y_pred = MLP.fit(test_data[["x"]].to_numpy())
    print(r2_score(test_data[["y"]], y_pred))
    # axs[0][j].scatter(mesh[:, 0], mesh[:, 1], s=0.5, c=pred_mesh)
    axs[j].scatter(test_data.x, test_data.y)
    axs[j].scatter(test_data.x, y_pred)

# for j, size in enumerate(sizes):
#     training_data = pd.read_csv(os.path.join(PATH, f"{datasets[1]}.train.{size}.csv"))
#     test_data = pd.read_csv(os.path.join(PATH, f"{datasets[1]}.test.{size}.csv"))
#     MLP = Network([1, 2, 1], regression=True, batch_size=10, n_epochs=50)
#     MLP.train_batches(training_data[["x"]].to_numpy(), training_data[["y"]].to_numpy())
#     y_pred = MLP.fit(test_data[["x"]].to_numpy())
#     print(r2_score(test_data[["y"]], y_pred))
#     axs[1][j].scatter(test_data.x, test_data.y)
#     axs[1][j].scatter(test_data.x, y_pred)

plt.show()
