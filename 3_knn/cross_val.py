import numpy as np
import typing
from collections import defaultdict


def kfold_split(num_objects: int,
                num_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split [0, 1, ..., num_objects - 1] into equal num_folds folds
       (last fold can be longer) and returns num_folds train-val
       pairs of indexes.

    Parameters:
    num_objects: number of objects in train set
    num_folds: number of folds for cross-validation split

    Returns:
    list of length num_folds, where i-th element of list
    contains tuple of 2 numpy arrays, he 1st numpy array
    contains all indexes without i-th fold while the 2nd
    one contains i-th fold
    """
    i = 0
    folds = []
    div, mod = divmod(num_objects, num_folds)
    num_objects = [i for i in range(num_objects)]
    while i < len(num_objects):
        folds.append((num_objects[i:i+div], i))
        i = i + div
    res = []
    for elem in folds:
        if mod != 0 and len(res) == num_folds - 1:
            break
        temp = []
        for i in range(len(num_objects)):
            if abs(i - elem[1]) < div:
                if i < elem[1]:
                    temp.append((num_objects[i]))
            else:
                temp.append((num_objects[i]))
        res.append((np.array(temp), np.array(elem[0])))
    if mod != 0:
        res.append((np.array(num_objects[:(num_folds-1)*div]), np.array(num_objects[(num_folds-1)*div:])))
    return res


def knn_cv_score(X: np.ndarray, y: np.ndarray, parameters: dict[str, list],
                 score_function: callable,
                 folds: list[tuple[np.ndarray, np.ndarray]],
                 knn_class: object) -> dict[str, float]:
    """Takes train data, counts cross-validation score over
    grid of parameters (all possible parameters combinations)

    Parameters:
    X: train set
    y: train labels
    parameters: dict with keys from
        {n_neighbors, metrics, weights, normalizers}, values of type list,
        parameters['normalizers'] contains tuples (normalizer, normalizer_name)
        see parameters example in your jupyter notebook

    score_function: function with input (y_true, y_predict)
        which outputs score metric
    folds: output of kfold_split
    knn_class: class of knn model to fit

    Returns:
    dict: key - tuple of (normalizer_name, n_neighbors, metric, weight),
    value - mean score over all folds
    """
    res = {}
    for n, m, w, normalize in [
                            (n, m, w, normalize) for n in parameters["n_neighbors"]
                            for m in parameters["metrics"]
                            for w in parameters["weights"]
                            for normalize in parameters["normalizers"]]:
        score = 0
        for item in folds:
            X_train = X[item[0]]
            X_val = X[item[1]]
            y_train = y[item[0]]
            y_val = y[item[1]]
            if normalize[0]:
                normalize[0].fit(X_train)
                X_train = normalize[0].transform(X_train)
                X_val = normalize[0].transform(X_val)
            model = knn_class(n_neighbors=n, metric=m, weights=w)
            model.fit(X=X_train, y=y_train)
            temp = model.predict(X_val)
            score += score_function(y_val, temp)
        score /= len(folds)
        res[(normalize[1], n, m, w)] = score
    return res
