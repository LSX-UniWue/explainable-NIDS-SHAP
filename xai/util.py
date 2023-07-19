
import numpy as np
import pandas as pd


def tabular_reference_points(background, X_expl, X_train=None, predict_fn=None):

    if background in ['mean', 'NN']:
        assert X_train is not None, f"background '{background}' requires train data as input at variable 'X_train'"
    if background in ['optimized']:
        assert predict_fn is not None, f"background '{background}' requires predict_fn as input"

    if background == 'zeros':  # zero vector, default
        reference_points = np.zeros(X_expl.shape)
        return reference_points

    elif background == 'mean':  # mean training data point for each data point
        reference_points = np.mean(X_train, axis=0).reshape((1, -1)).repeat(X_expl.shape[0], axis=0)
        return reference_points

    elif background == 'NN':  # nearest neighbor in the normal training data for each data point
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=1).fit(X_train)
        neighbor_dist, neighbor_idx = nbrs.kneighbors(X=X_expl, n_neighbors=1, return_distance=True)
        reference_points = X_train[neighbor_idx.flatten()]
        return reference_points

    elif background == 'kmeans':  # kmeans cluster centers of normal data as global background
        from sklearn.cluster import k_means
        centers, _, _ = k_means(X=X_train, n_clusters=5)
        return centers

    else:
        raise ValueError(f"Unknown background: {background}")
