import numpy as np


def entropy(data, num_bins=120, decimals=4):
    if len(data) == 0:
        return 0
    counts, bins = np.histogram(data)
    P = counts / np.sum(counts)
    # filter out zero probabilities
    P = P[P > 0]
    return np.around(-np.sum(P * np.log2(P)), decimals=decimals)


def gini_index(data, num_bins=120, decimals=4):
    if len(data) == 0:
        return 0
    counts, bins = np.histogram(data)
    P = counts / np.sum(counts)
    return np.around(1 - np.sum(P ** 2), decimals=decimals)
