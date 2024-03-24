import numpy as np

from numpy.testing import assert_equal, assert_allclose
from sklearn import neighbors
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from cross_val import kfold_split, knn_cv_score


def test_split_0():
    with open('cross_val.py', 'r') as file:
        lines = ' '.join(file.readlines())
        assert 'import numpy' in lines
        assert 'import defaultdict' in lines
        assert 'import typing' in lines
        assert lines.count('import') == 3
        assert 'sklearn' not in lines

def test_split_1():
    X_1 = kfold_split(2, 2)
    answer = [(np.array([1]), np.array([0])), (np.array([0]), np.array([1]))]

    assert type(X_1) == list
    assert_equal(X_1, answer)

def test_split_2():
    X_1 = kfold_split(5, 3)
    answer = [(np.array([1, 2, 3, 4]), np.array([0])),
              (np.array([0, 2, 3, 4]), np.array([1])),
              (np.array([0, 1]), np.array([2, 3, 4]))]

    assert type(X_1) == list
    assert_equal(X_1, answer)
    
def test_split_3():
    X_1 = kfold_split(11, 7)
    answer = [(np.array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]), np.array([0])),
              (np.array([ 0,  2,  3,  4,  5,  6,  7,  8,  9, 10]), np.array([1])),
              (np.array([ 0,  1,  3,  4,  5,  6,  7,  8,  9, 10]), np.array([2])),
              (np.array([ 0,  1,  2,  4,  5,  6,  7,  8,  9, 10]), np.array([3])),
              (np.array([ 0,  1,  2,  3,  5,  6,  7,  8,  9, 10]), np.array([4])),
              (np.array([ 0,  1,  2,  3,  4,  6,  7,  8,  9, 10]), np.array([5])),
              (np.array([0, 1, 2, 3, 4, 5]), np.array([ 6,  7,  8,  9, 10]))]

    assert type(X_1) == list
    assert_equal(X_1, answer)

def test_cv_4():
    X_train = np.array([[2, 1, -1], [1, 1, 1], [0.9, -0.25, 7], [1, 2, -3], [0, 0, 0], [2, -1, 0.5]])
    y_train = np.sum(X_train, axis=1)
    parameters = {
        'n_neighbors': [1, 2, 4],
        'metrics': ['euclidean', 'cosine'],
        'weights': ['uniform', 'distance'],
        'normalizers': [(None, 'None')]
    }
    folds = kfold_split(6, 3)
    out = knn_cv_score(X_train, y_train, parameters, r2_score, folds, neighbors.KNeighborsRegressor)

    answer = {
        ('None', 1, 'euclidean', 'uniform'): -11.29188203967135,
        ('None', 1, 'euclidean', 'distance'): -11.29188203967135,
        ('None', 1, 'cosine', 'uniform'): -17.63280796559728,
        ('None', 1, 'cosine', 'distance'): -17.632807965597276,
        ('None', 2, 'euclidean', 'uniform'): -7.5333863756105215,
        ('None', 2, 'euclidean', 'distance'): -7.997305328982919,
        ('None', 2, 'cosine', 'uniform'): -4.246942433109774,
        ('None', 2, 'cosine', 'distance'): -6.7448165099645365,
        ('None', 4, 'euclidean', 'uniform'): -3.6194607932134364,
        ('None', 4, 'euclidean', 'distance'): -4.211377791660151,
        ('None', 4, 'cosine', 'uniform'): -3.6194607932134364,
        ('None', 4, 'cosine', 'distance'): -4.335691384752842
     }

    assert type(out) == dict
    assert len(out) == len(answer)
    for key in answer:
        assert_allclose(answer[key], out[key])

def test_cv_5():
    X_train = np.array([[ 0.62069296, -0.07097426,  0.65172896, -1.14620331],
                        [ 2.03347616,  0.32524614, -0.71941433, -0.30789854],
                        [ 0.17100377,  1.63120292,  1.34284446, -2.16397238],
                        [-1.65370417,  0.62499229, -0.50217293,  2.07813591],
                        [ 0.84667916,  0.25458428,  0.14720704, -0.18668345],
                        [ 0.43833344, -1.40348048, -1.37944118,  0.19192659],
                        [ 0.97229574, -0.54606276, -0.09855294,  1.28961291],
                        [ 0.25355626, -1.72816511,  0.084554  , -2.14256875],
                        [ 0.36103462, -1.28930935,  1.34586369, -0.57300728],
                        [-1.42711933, -0.11832827, -0.58038295, -1.56806583]])
    y_train = np.sum(np.abs(X_train), axis=1)
    parameters = {
        'n_neighbors': [1, 2, 4],
        'metrics': ['euclidean', 'cosine'],
        'weights': ['uniform', 'distance'],
        'normalizers': [(None, 'None')]
    }
    folds = kfold_split(10, 3)
    out = knn_cv_score(X_train, y_train, parameters, r2_score, folds, neighbors.KNeighborsRegressor)

    answer = {
        ('None', 1, 'euclidean', 'uniform'): -3.8869140469579033,
        ('None', 1, 'euclidean', 'distance'): -3.8869140469579033,
        ('None', 1, 'cosine', 'uniform'): -3.8967543637841557,
        ('None', 1, 'cosine', 'distance'): -3.8967543637841557,
        ('None', 2, 'euclidean', 'uniform'): -2.8537893891104353,
        ('None', 2, 'euclidean', 'distance'): -2.8723718210868676,
        ('None', 2, 'cosine', 'uniform'): -0.9110922868244854,
        ('None', 2, 'cosine', 'distance'): -1.2935713889809644,
        ('None', 4, 'euclidean', 'uniform'): -1.0722930212962776,
        ('None', 4, 'euclidean', 'distance'): -1.291339953080277,
        ('None', 4, 'cosine', 'uniform'): 0.010193326582544423,
        ('None', 4, 'cosine', 'distance'): -0.38359677639174855
     }

    assert type(out) == dict
    assert len(out) == len(answer)
    for key in answer:
        assert_allclose(answer[key], out[key])

def test_cv_6():
    X_train = np.array([[ 0.62069296, -0.07097426,  0.65172896, -1.14620331],
                        [ 2.03347616,  0.32524614, -0.71941433, -0.30789854],
                        [ 0.17100377,  1.63120292,  1.34284446, -2.16397238],
                        [-1.65370417,  0.62499229, -0.50217293,  2.07813591],
                        [ 0.84667916,  0.25458428,  0.14720704, -0.18668345],
                        [ 0.43833344, -1.40348048, -1.37944118,  0.19192659],
                        [ 0.97229574, -0.54606276, -0.09855294,  1.28961291],
                        [ 0.25355626, -1.72816511,  0.084554  , -2.14256875],
                        [ 0.36103462, -1.28930935,  1.34586369, -0.57300728],
                        [-1.42711933, -0.11832827, -0.58038295, -1.56806583]])
    y_train = np.sum(np.abs(X_train), axis=1)
    scaler = MinMaxScaler()
    parameters = {
        'n_neighbors': [1, 2, 4],
        'metrics': ['euclidean', 'cosine'],
        'weights': ['uniform', 'distance'],
        'normalizers': [(scaler, 'MinMaxScaler')]
    }
    folds = kfold_split(10, 3)
    out = knn_cv_score(X_train, y_train, parameters, r2_score, folds, neighbors.KNeighborsRegressor)

    answer = {
        ('MinMaxScaler', 1, 'euclidean', 'uniform'): -3.886914104013526,
        ('MinMaxScaler', 1, 'euclidean', 'distance'): -3.886914104013526,
        ('MinMaxScaler', 1, 'cosine', 'uniform'): -3.339427669385479,
        ('MinMaxScaler', 1, 'cosine', 'distance'): -3.339427669385479,
        ('MinMaxScaler', 2, 'euclidean', 'uniform'): -2.821522070818421,
        ('MinMaxScaler', 2, 'euclidean', 'distance'): -2.909515284414977,
        ('MinMaxScaler', 2, 'cosine', 'uniform'): -3.0373126577073877,
        ('MinMaxScaler', 2, 'cosine', 'distance'): -2.7197802024893374,
        ('MinMaxScaler', 4, 'euclidean', 'uniform'): -1.229118435031323,
        ('MinMaxScaler', 4, 'euclidean', 'distance'): -1.4848798788742938,
        ('MinMaxScaler', 4, 'cosine', 'uniform'): -0.3586698577110674,
        ('MinMaxScaler', 4, 'cosine', 'distance'): -0.7914850319051477
     }

    assert type(out) == dict
    assert len(out) == len(answer)
    for key in answer:
        assert_allclose(answer[key], out[key])
