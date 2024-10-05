import numpy as np


def entropy(data, num_bins=120, decimals=4):
    # Get the unique classes and their counts
    classes, counts = np.unique(data, return_counts=True)

    # Calculate the total number of samples in the node
    total_samples = np.sum(counts)

    # Calculate the probabilities of each class
    probabilities = counts / total_samples

    # Compute entropy: -sum(p_i * log2(p_i))
    entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])  # Avoid log(0) issues
    return entropy


def gini_index(data, num_bins=120, decimals=4):
    # Get the unique classes and their counts
    classes, counts = np.unique(data, return_counts=True)

    # Calculate the total number of samples in the node
    total_samples = np.sum(counts)

    # Calculate the probabilities of each class
    probabilities = counts / total_samples

    # Compute Gini impurity: 1 - sum(p_i^2)
    gini = 1 - np.sum(probabilities ** 2)
    return gini