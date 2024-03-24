import numpy as np


def evaluate_measures(sample):
    _, counts = np.unique(sample, return_counts=True)
    p = counts / sum(counts)
    gini = 1 - sum(p ** 2)
    entropy = - sum(p * np.log(p))
    error = 1 - max(p)
    measures = {'gini': float(gini), 'entropy': float(entropy), 'error': float(error)}
    return measures
