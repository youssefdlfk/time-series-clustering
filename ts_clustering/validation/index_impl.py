import numpy as np
from sklearn.metrics import euclidean_distances, pairwise_distances
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.metrics import cdist_dtw, dtw

from ts_clustering.clustering.utils import (compute_WCSS,
                                            cross_correlation_average,
                                            distance_cross_correlation,
                                            pairwise_cross_correlation)


def silhouette_index(X: np.ndarray, labels: np.ndarray, distance_matrix: np.ndarray) -> float:
    """
    Compute Silhouette index for a time series clustering. The higher the better. Within [-1, 1].
    :param distance_matrix: Distance matrix
    :param X: Tensor of time series data
    :param labels: Array of the cluster each data point belongs to
    :return: Silhouette index
    """
    n_samples = X.shape[0]
    n_clusters = len(np.unique(labels))
    # Initialize arrays to store a and b values
    a = np.zeros(n_samples)
    b = np.full(n_samples, np.inf)
    # For each cluster, compute a and b
    for k in range(n_clusters):
        # Indices of samples in cluster k
        cluster_k_indices = np.where(labels == k)[0]
        # Compute intra-cluster distances (a)
        if len(cluster_k_indices) > 1:
            intra_distances = distance_matrix[np.ix_(cluster_k_indices, cluster_k_indices)]
            # Exclude self-distances by setting the diagonal to NaN
            np.fill_diagonal(intra_distances, np.nan)
            a[cluster_k_indices] = np.nanmean(intra_distances, axis=1)
        else:
            # For clusters with one sample, a is zero
            a[cluster_k_indices] = 0
        # Compute nearest-cluster distances (b)
        for l in range(n_clusters):
            if k != l:
                cluster_l_indices = np.where(labels == l)[0]
                inter_distances = distance_matrix[np.ix_(cluster_k_indices, cluster_l_indices)]
                mean_inter_distances = np.mean(inter_distances, axis=1)
                b[cluster_k_indices] = np.minimum(b[cluster_k_indices], mean_inter_distances)
    # Compute the silhouette scores
    s = (b - a) / np.maximum(a, b)
    # Handle cases where a and b are zero
    s[np.isnan(s)] = 0
    return np.mean(s)


def dunn_index(labels: np.ndarray, distance_matrix: np.ndarray) -> float:
    """
    Compute the Dunn index of a time series clustering with optimized computations. The higher the better.
    """
    n_clusters = len(np.unique(labels))

    # Initialize lists to store intra-cluster and inter-cluster distances
    intra_dists = []
    inter_dists = []
    # Compute maximum intra-cluster distances
    for k in range(n_clusters):
        cluster_indices = np.where(labels == k)[0]
        if len(cluster_indices) > 1:
            intra_cluster_dist = distance_matrix[np.ix_(cluster_indices, cluster_indices)]
            np.fill_diagonal(intra_cluster_dist, np.nan)
            intra_dists.append(np.nanmax(intra_cluster_dist))
        else:
            intra_dists.append(0)

    # Compute minimum inter-cluster distances
    for i in range(n_clusters):
        cluster_i_indices = np.where(labels == i)[0]
        for j in range(i + 1, n_clusters):
            cluster_j_indices = np.where(labels == j)[0]
            inter_cluster_dist = distance_matrix[np.ix_(cluster_i_indices, cluster_j_indices)]
            inter_dists.append(np.min(inter_cluster_dist))

    numerator = np.min(inter_dists)
    denominator = np.max(intra_dists)
    dunn_index = numerator / denominator if denominator != 0 else 0
    return dunn_index


def davies_bouldin_index(X: np.ndarray, model, labels: np.ndarray, metric: str, **kwargs) -> float:
    """
    Compute the Davies-Bouldin index for a time series clustering. The lower the better.
    :param model: Clustering model
    :param X: Torch tensor of shape [N, d], N = number of time series, d = number of samples for each
    :param labels: Numpy array of the cluster number for each data point
    :param metric: The similarity measure (euclidean, dtw or cross-correlation)
    :param kwargs: Additional constraint parameters for DTW metric (optional)
    :return: Davies-Bouldin index
    """
    # n_clusters is the number of clusters
    n_clusters = model.n_clusters
    # cluster_k is a list of indices of data points in cluster k
    clusters = [X[labels == k] for k in range(n_clusters)]
    # get centroids as 2D (1, T) for pairwise routines
    centroids = [np.expand_dims(model.cluster_centers_[k].squeeze(-1), axis=0) for k in range(n_clusters)]
    # stack centroids as (k, T) for pairwise centroid–centroid distances
    stacked_centroids = np.vstack([c for c in centroids])  # shape: (k, T)

    # --- Δ(X_i): intracluster scatter for each cluster i ---
    Delta = np.zeros(n_clusters)
    for i in range(n_clusters):
        if metric == 'euclidean':
            # mean distance of points to centroid i
            Delta[i] = np.mean(pairwise_distances(clusters[i], centroids[i]))
        elif metric == 'dtw':
            Delta[i] = np.mean(
                cdist_dtw(clusters[i], centroids[i],
                          global_constraint=kwargs['metric_params']['global_constraint'],
                          sakoe_chiba_radius=kwargs['metric_params']['sakoe_chiba_radius'],
                          itakura_max_slope=kwargs['metric_params']['itakura_max_slope'])
            )
        elif metric == 'cross-correlation':
            Delta[i] = np.mean(pairwise_cross_correlation(clusters[i], centroids[i], self_similarity=False))
        else:
            raise ValueError('Unsupported metric.')

    # --- δ(X_i, X_j): centroid–centroid distances for all pairs (i, j) ---
    if metric == 'euclidean':
        delta = euclidean_distances(stacked_centroids, stacked_centroids)  # shape (k, k)
    elif metric == 'dtw':
        delta = cdist_dtw(stacked_centroids, stacked_centroids,
                          global_constraint=kwargs['metric_params']['global_constraint'],
                          sakoe_chiba_radius=kwargs['metric_params']['sakoe_chiba_radius'],
                          itakura_max_slope=kwargs['metric_params']['itakura_max_slope'])
    elif metric == 'cross-correlation':
        delta = pairwise_cross_correlation(stacked_centroids, stacked_centroids, self_similarity=True)
    else:
        raise ValueError('Unsupported metric.')

    # --- S_ij = (Δ_i + Δ_j) / δ_ij for all i != j; DB = mean_i max_j S_ij ---
    S = (Delta[:, None] + Delta[None, :]) / delta  # shape (k, k)
    np.fill_diagonal(S, -np.inf)  # exclude j == i for the max
    DB = np.mean(np.max(S, axis=1))
    return DB


def stability_index(X: np.ndarray, labels: np.ndarray, labels_cut: np.ndarray, distance_matrix: np.ndarray, method: str) -> float:
    """
    Compute stability measures for time series clustering by deleting a percentage of the dataset columns and comparing the
    perturbed clustering with the unperturbed one. The lower, the better.
    :param method: Name of specific method used
    :param labels_cut:
    :param distance_matrix:
    :param X:
    :param model:
    :param labels:
    :param kwargs:
    :return:

    Reference: Alboukadel Kassambara (2017). "Practical Guide To Cluster Analysis in R".
    """
    n_clusters = len(np.unique(labels))
    # Compute Average Distance (AD) stability measure
    # Correspond to the average distance between observations placed in the same cluster under both cases (the lower the better)
    if method == 'ad':
        # Compute the AD index
        ad = 0
        total_oij = 0
        for i in range(n_clusters):
            cluster_i_indices = np.where(labels == i)[0]
            for j in range(n_clusters):
                cluster_j_indices_cut = np.where(labels_cut == j)[0]
                oij = np.intersect1d(cluster_i_indices, cluster_j_indices_cut).size
                if oij > 0:
                    dij = np.mean(distance_matrix[np.ix_(cluster_i_indices, cluster_j_indices_cut)])
                    ad += oij * dij
                    total_oij += oij
        return ad/X.shape[0]

    # Compute Average Proportion of Non-overlap (APN) stability measure
    # Correspond to the proportion observations not placed in the same cluster after perturbation (the lower the better)
    elif method == 'apn':
        # Compute the Average Proportion of Non-overlap (APN)
        total_oij = 0
        apn_sum = 0
        for i in range(n_clusters):
            cluster_indices = np.where(labels == i)[0]
            labels_cut_cluster = labels_cut[cluster_indices]
            counts = np.bincount(labels_cut_cluster, minlength=n_clusters)
            total_in_cluster = len(cluster_indices)
            apn_sum += np.sum(counts ** 2) / total_in_cluster
            total_oij += np.sum(counts)
        apn = 1 - (apn_sum / total_oij)
        return apn

    else:
        raise ValueError('Unsupported method.')


def calinski_harabasz_index(X: np.ndarray, model, labels: np.ndarray, metric: str, **kwargs) -> float:
    """
    Compute the Calinski-Harabasz index for time series clustering, i.e. a ratio of between-cluster dispersion (B)
    and within-cluster dispersion (W). The higher the value, the better the clustering.
    :param model: Clustering model
    :param X: Torch tensor of shape [N, d], N = number of time series, d = number of samples for each
    :param labels: Numpy array of the cluster number for each data point
    :param metric: The similarity measure (euclidean, dtw or cross-correlation)
    :param kwargs: Additional constraint parameters for DTW metric (optional)
    :return: Calinski-Harabasz index
    """
    # n_clusters is the number of clusters
    n_clusters = model.n_clusters
    # cluster_k is a list of indices of data points in cluster k
    cluster_k = [X[labels == k] for k in range(n_clusters)]
    # factor
    scaling_factor = (len(X) - n_clusters)/(n_clusters-1)
    # between cluster dispersion
    B = 0
    # centroids is an array of the cluster centroids
    centroids = [model.cluster_centers_.squeeze(-1)[k] for k in range(n_clusters)]
    if metric == 'euclidean':
        # compute mean of centroids
        mu = np.mean(np.vstack(centroids), axis=0)
        for i in range(n_clusters):
            n_i = cluster_k[i].shape[0]
            mu_i = centroids[i]
            B += n_i * np.sum((mu_i - mu) ** 2)
    elif metric == 'dtw':
        mu = dtw_barycenter_averaging(np.vstack(centroids)).squeeze(-1)
        for i in range(n_clusters):
            n_i = cluster_k[i].shape[0]
            mu_i = centroids[i]
            B += n_i * (dtw(mu_i, mu, global_constraint=kwargs['metric_params']['global_constraint'],
                            sakoe_chiba_radius=kwargs['metric_params']['sakoe_chiba_radius'],
                            itakura_max_slope=kwargs['metric_params']['itakura_max_slope']) ** 2)
    elif metric == 'cross-correlation':
        mu = cross_correlation_average(np.expand_dims(np.vstack(centroids), axis=-1)).squeeze(-1)
        for i in range(n_clusters):
            n_i = cluster_k[i].shape[0]
            mu_i = centroids[i]
            B += n_i * (distance_cross_correlation(mu_i, mu) ** 2)

    # within-cluster dispersion
    W = compute_WCSS(n_clusters=n_clusters, cluster_k=cluster_k, centroids=centroids, metric=metric, **kwargs)
    return (B/W)*scaling_factor


def hartigan_index(X: np.ndarray, model_k, labels_k: np.ndarray, model_k1,
                   labels_k1: np.ndarray, metric: str, **kwargs) -> float:
    """
    Compute the Hartigan validity index for time series clustering, i.e. the ratio of within-cluster sum of squares for
    cluster k and for cluster k + 1
    The lower the better
    :param labels_k1: Clustering labels for k + 1 clusters
    :param model_k1: Clustering model for k + 1 clusters
    :param labels_k: Clustering labels for k clusters
    :param model_k: Clustering model for k clusters
    :param X: Torch tensor of shape [N, d], N = number of time series, d = number of samples for each
    :param labels: Numpy array of the cluster number for each data point
    :param metric: The similarity measure (euclidean, dtw or cross-correlation)
    :param kwargs: Additional constraint parameters for DTW metric (optional)
    :return: Hartigan index

    Reference: Natacha Galmiche (2024). "PyCVI: A Python package for internal Cluster Validity Indices,
    compatible with time-series data"
    """
    # Compute WCSS1 for model with k clusters
    n_clusters1 = model_k.n_clusters
    # cluster_k is a list of indices of data points in cluster k
    cluster_k1 = [X[labels_k == k] for k in range(n_clusters1)]
    centroids1 = [model_k.cluster_centers_.squeeze(-1)[k] for k in range(n_clusters1)]
    WCSS1 = compute_WCSS(n_clusters=n_clusters1, cluster_k=cluster_k1, centroids=centroids1, metric=metric, **kwargs)

    # Compute WCSS2 for model with k+1 clusters
    n_clusters2 = model_k1.n_clusters
    # cluster_k2 is a list of indices of data points in cluster k+1
    cluster_k2 = [X[labels_k1 == k] for k in range(n_clusters2)]
    centroids2 = [model_k1.cluster_centers_.squeeze(-1)[k] for k in range(n_clusters2)]
    WCSS2 = compute_WCSS(n_clusters=n_clusters2, cluster_k=cluster_k2, centroids=centroids2, metric=metric, **kwargs)
    # Multiplying factor
    factor = X.shape[0] - n_clusters1 - 1
    return (WCSS1/WCSS2 - 1)*factor