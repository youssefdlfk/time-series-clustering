import numpy as np
from dtaidistance.dtw import distance_matrix_fast
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import pairwise_distances
from tslearn.metrics import (cdist_dtw, cdist_normalized_cc, dtw,
                             y_shifted_sbd_vec)
from tslearn.metrics.cycc import normalized_cc


def compute_distance_matrix(X: np.ndarray, metric: str, **kwargs) -> np.ndarray:
    """
    Compute distance matrix for each metric,
    :param X:
    :param metric:
    :param kwargs:
    :return:
    """
    n_samples = X.shape[0]
    if metric == 'euclidean':
        # distance_matrix = pairwise_distances(X.reshape(n_samples, -1), metric='euclidean', n_jobs=-1)
        distance_matrix = squareform(pdist(X.reshape(n_samples, -1), metric='euclidean'))
    elif metric == 'dtw':
        distance_matrix = cdist_dtw(X, global_constraint=kwargs['metric_params']['global_constraint'],
                                   sakoe_chiba_radius=kwargs['metric_params']['sakoe_chiba_radius'],
                                   itakura_max_slope=kwargs['metric_params']['itakura_max_slope'], n_jobs=-1)
        # distance_matrix = distance_matrix_fast(X, window=kwargs['metric_params']['sakoe_chiba_radius'], use_c=True,
        #                                            parallel=True)
    elif metric == 'cross-correlation':
        distance_matrix = pairwise_cross_correlation(X, X, self_similarity=True)
    else:
        raise ValueError('Unsupported metric.')
    return distance_matrix


def compute_WCSS(n_clusters: int, cluster_k: list[np.ndarray], centroids, metric, **kwargs):
    """
    Compute the Within-Cluster Sum Squared distance
    :param n_clusters: Number of clusters
    :param cluster_k: List of time series clusters
    :param centroids: List of centroids
    :param metric: Metric distance
    :param kwargs: Additional metric parameters
    :return: The Within-Cluster Sum of Squared distances
    """
    WCSS = 0.0
    for i in range(n_clusters):
        variance = 0.0
        for datapoint in cluster_k[i]:
            if metric == 'euclidean':
                variance += np.sum((datapoint - centroids[i])**2)
            elif metric == 'dtw':
                variance += dtw(datapoint, centroids[i], global_constraint=kwargs['metric_params']['global_constraint'],
                                sakoe_chiba_radius=kwargs['metric_params']['sakoe_chiba_radius'],
                                itakura_max_slope=kwargs['metric_params']['itakura_max_slope'])**2
            elif metric == 'cross-correlation':
                variance += distance_cross_correlation(datapoint, centroids[i])**2
        WCSS += variance
    return WCSS


def clusters_labels_to_indices(labels: np.ndarray):
    """
    Convert the cluster labels array (values = index of the cluster each data point belongs to) to clusters indices array
    (values = index of the data point within the list corresponding to the cluster it belongs to)
    :param labels: Cluster labels array
    :return: Cluster indices array
    """
    #indices_all = []
    #for cluster_idx in np.unique(labels):
    #    cluster_idx_list = []
    #    for data_idx in range(len(labels)):
    #        if labels[data_idx] == cluster_idx:
    #            cluster_idx_list.append(data_idx)
    #    indices_all.append(cluster_idx_list)
    indices_all = [np.where(labels == cluster_idx)[0].tolist() for cluster_idx in np.unique(labels)]
    return indices_all


def distance_cross_correlation(X1: np.ndarray, X2: np.ndarray):
    """
    Compute cross-correlation between two time series
    :param X1: First time series
    :param X2: Second time series
    :return: Cross correlation measure
    """
    return normalized_cc(np.expand_dims(X1, axis=-1), np.expand_dims(X2, axis=-1)).max()


def pairwise_cross_correlation(X1: np.ndarray, X2: np.ndarray, self_similarity):
    """
    Compute cross-correlation matrix between two sets of time series
    :param X1: First dataset
    :param X2: Second dataset
    :return: Normalized pairwise cross correlation measure
    """
    # Zero‑lag normalized cross‑corr
    # Xc = X - X.mean(axis=1, keepdims=True)
    # norms = np.linalg.norm(Xc, axis=1, keepdims=True)
    # Xn = Xc / norms
    # sim = Xn @ Xn.T  # very fast
    # dist = 1 - sim
    return cdist_normalized_cc(np.expand_dims(X1, axis=-1), np.expand_dims(X2, axis=-1), np.ones(X1.shape[0])*-1,
                               np.ones(X2.shape[0])*-1, self_similarity)


def cross_correlation_average(dataset, max_iters=10, tol=1e-4):
    """
    Compute the global average (centroid) of a time series dataset, considering cross-correlation.

    Parameters
    ----------
    dataset : array-like, shape=(n_ts, sz, d), dtype=float64
        A dataset of time series.
    max_iters : int, default=10
        Maximum number of iterations to compute the centroid.
    tol : float, default=1e-4
        Convergence tolerance for stopping criterion.

    Returns
    -------
    global_centroid : array-like, shape=(sz, d), dtype=float64
        The computed global average time series (centroid).
    """
    n_ts, sz, d = dataset.shape

    # Initialize the centroid (using the first time series as a reference)
    global_centroid = dataset[0].copy()

    # Initialize norms for the dataset and the reference
    norms_dataset = np.array([np.linalg.norm(dataset[i]) for i in range(n_ts)], dtype=np.float64)
    norm_ref = np.linalg.norm(global_centroid)

    for _ in range(max_iters):
        # Shift all time series in the dataset relative to the current global centroid
        shifted_dataset = y_shifted_sbd_vec(global_centroid, dataset, norm_ref, norms_dataset)

        # Update the global centroid as the mean of the aligned time series
        new_centroid = np.mean(shifted_dataset, axis=0)

        # Check for convergence
        if np.linalg.norm(new_centroid - global_centroid) < tol:
            break

        # Update centroid and norm for the next iteration
        global_centroid = new_centroid
        norm_ref = np.linalg.norm(global_centroid)

    return global_centroid


def spearman_footrule_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute the Spearman footrule distance between two vectors
    :param X: First vector
    :param Y: Second vector
    :return: Spearman footrule distance
    """
    d = len(X)
    if d % 2 == 0:
        max_d = (d-1)*d/2
    elif d % 2 != 0:
        max_d = (d**2)/2
    dist_XY = 0
    for i in range(d):
        dist_XY += abs(X[i]-Y[i])
    return dist_XY/max_d
