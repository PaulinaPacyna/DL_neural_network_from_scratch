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
size = sizes[2]
training_data = pd.read_csv(os.path.join(PATH, f"{datasets[0]}.train.{size}.csv"))
# training_data["x"] = (training_data["x"] - min(training_data["x"])) / (
#     max(training_data["x"]) - min(training_data["x"])
# )
# training_data["y"] = (training_data["y"] - min(training_data["y"])) / (
#     max(training_data["y"]) - min(training_data["y"])
# )
test_data = pd.read_csv(os.path.join(PATH, f"{datasets[0]}.test.{size}.csv"))
# test_data["x"] = (test_data["x"] - min(test_data["x"])) / (
#     max(test_data["x"]) - min(test_data["x"])
# )
# test_data["y"] = (test_data["y"] - min(test_data["y"])) / (
#     max(test_data["y"]) - min(test_data["y"])
# )
MLP = Network(
    [1, 2, 2, 1],
    regression=True,
    n_epochs=10000,
    batch_size=100,
    activation_type="sigmoid",
    learning_rate=0.01,
    momentum_rate=0.01,
    print_progress=True,
    cost_fun="cross-entropy",
)
MLP.train(training_data[["x"]].to_numpy(), training_data[["y"]].to_numpy())
y_pred = MLP.fit(test_data[["x"]].to_numpy(), predict=True)
print(r2_score(test_data[["y"]], y_pred))
plt.scatter(test_data[["x"]], test_data[["y"]], c="red")
plt.scatter(
    training_data[["x"]], training_data[["y"]], c="green"
)  # jak przyblizysz to widać
plt.scatter(test_data[["x"]], y_pred, c="blue")
plt.show()
