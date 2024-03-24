import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
import numpy as np


def recenter(arr):
    import scipy as sp
    slicing = sp.ndimage.find_objects(arr != 0, max_label=1)[0]
    center_slicing = tuple(
        slice((dim - sl.stop + sl.start) // 2, (dim + sl.stop - sl.start) // 2)
        for sl, dim in zip(slicing, arr.shape))
    result = np.zeros_like(arr)
    result[center_slicing] = arr[slicing]
    return result


class PotentialTransformer:
    def fit(self, x, y):
        return self

    def fit_transform(self, x, y):
        return self.transform(x)

    def transform(self, x):
        for idx, _ in enumerate(x):
            x[idx] -= 20
            x[idx] = recenter(x[idx])
        x_train = np.array([np.array(list(map(sum, zip(*elem)))) for elem in x])
        return x_train


def load_dataset(data_dir):
    files, X, Y = [], [], []
    for file in sorted(os.listdir(data_dir)):
        potential = np.load(os.path.join(data_dir, file))
        files.append(file)
        X.append(potential["data"])
        Y.append(potential["target"])
    return files, np.array(X), np.array(Y)


def train_model_and_predict(train_dir, test_dir):
    _, X_train, Y_train = load_dataset(train_dir)
    test_files, X_test, _ = load_dataset(test_dir)
    p = PotentialTransformer()
    X_train = p.fit_transform(X_train, Y_train)
    X_test = p.fit_transform(X_test, Y_train)
    regressor = ExtraTreesRegressor(max_features="sqrt", max_depth=10, criterion="friedman_mse", n_estimators=3000)
    regressor.fit(X_train, Y_train)
    predictions = regressor.predict(X_test)
    return {file: value for file, value in zip(test_files, predictions)}
