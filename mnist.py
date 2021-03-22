import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pickle

from network import Network

try:  # for slightly faster loading
    train = pickle.load('train.pkl')
    test = pickle.load('test.pkl')
except TypeError:
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    train.to_pickle('train.pkl')
    test.to_pickle('test.pkl')
print('Read')

X_test, X_train, y_test, y_train = train_test_split(train.drop(columns=['label']), train[['label']])
encoder = OneHotEncoder()
y_train = encoder.fit_transform(y_train).todense()
N = Network([X_train.shape[1], 64, 32, y_train.shape[1]], batch_size=50, n_epochs=1000, print_progress=True)
X_train /= 255
X_test /= 255
N.train(X_train, y_train)
prediction = encoder.inverse_transform(N.fit(X_test))
print((prediction == y_test).mean())
