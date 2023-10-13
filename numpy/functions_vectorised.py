import numpy as np
from typing import Tuple


def sum_non_neg_diag(X: np.ndarray) -> int:
    X = np.diag(X)
    X = X[X >= 0]
    return np.sum(X) if X.size > 0 else -1


def are_multisets_equal(x: np.ndarray, y: np.ndarray) -> bool:
    x = np.sort(x)
    y = np.sort(y)
    return np.array_equal(x, y)


def max_prod_mod_3(x: np.ndarray) -> int:
    y =  x[:-1] * x[1:]
    y = y[y % 3 == 0]
    return np.amax(y) if y.size > 0 else -1


def convert_image(image: np.ndarray, weights: np.ndarray) -> np.ndarray:
    res = image * weights
    res = np.sum(res, axis = -1)
    return res


def rle_scalar(x: np.ndarray, y: np.ndarray) -> int:
    x_el, x_cnt = x[:, 0], x[:, 1]
    y_el, y_cnt = y[:, 0], y[:, 1]
    if np.sum(x_cnt) != np.sum(y_cnt):
        return -1
    res = np.dot(np.repeat(x_el, x_cnt), np.repeat(y_el, y_cnt))
    return res


def cosine_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    norm_X = np.linalg.norm(X, axis=1, keepdims=True)
    norm_Y = np.linalg.norm(Y, axis=1, keepdims=True)
    prod = np.dot(X, Y.T)
    res = prod / (norm_X.dot(norm_Y.T))
    res[np.isnan(res)] = 1
    return res

