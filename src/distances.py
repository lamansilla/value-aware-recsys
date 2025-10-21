import numpy as np

import sklearn.metrics.pairwise as sm


def jaccard_distance(X):
    """
    Computes the Jaccard distance matrix between rows of X.

    Parameters
    ----------
    X : numpy.ndarray
        Input matrix of shape (n_samples, n_features).

    Returns
    -------
    numpy.ndarray
        Symmetric distance matrix of shape (n_samples, n_samples).
    """
    X_binary = (X > 0).astype(int)
    n_samples = X_binary.shape[0]
    dist_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            intersection = np.sum(X_binary[i] & X_binary[j])
            union = np.sum(X_binary[i] | X_binary[j])
            dist = 1.0 - (intersection / union) if union != 0 else 1.0
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    return dist_matrix


def euclidean_distance(X):
    """
    Computes the Euclidean distance matrix between rows of X.

    Parameters
    ----------
    X : numpy.ndarray
        Input matrix of shape (n_samples, n_features).

    Returns
    -------
    numpy.ndarray
        Symmetric distance matrix of shape (n_samples, n_samples).
    """
    return sm.euclidean_distances(X)


def madd_distance(X):
    """
    Computes the MADD distance matrix between rows of X.

    Parameters
    ----------
    X : numpy.ndarray
        Input matrix of shape (n_samples, n_features).

    Returns
    -------
    numpy.ndarray
        Symmetric distance matrix of shape (n_samples, n_samples).
    """
    n_samples, d = X.shape

    pair_dist = euclidean_distance(X) / np.sqrt(d)
    madd_dist = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            diff = np.abs(pair_dist[i] - pair_dist[j])
            mask = np.ones(n_samples, dtype=bool)
            mask[[i, j]] = False  # exclude i and j
            val = np.mean(diff[mask])
            madd_dist[i, j] = val
            madd_dist[j, i] = val

    return madd_dist
