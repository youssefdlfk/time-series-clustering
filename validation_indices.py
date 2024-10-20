import numpy as np
from sklearn.metrics import pairwise_distances, euclidean_distances
from tslearn.neighbors import KNeighborsTimeSeries
from aeon.distances import dtw_pairwise_distance, dtw_distance
from aeon.clustering.averaging import elastic_barycenter_average
import random
from utils_clustering import pairwise_cross_correlation, distance_cross_correlation, cross_correlation_average, compute_WCSS


def silhouette_index(X, model, labels, metric, **kwargs):
    """
    Compute Silhouette index for a time series clustering. The higher the better. Within [-1, 1].
    :param X: Tensor of time series data
    :param model: Clustering model from tslearn package
    :param labels: Array of the cluster each data point belongs to
    :param metric: Similarity measure (euclidean, dtw or cross-correlation)
    :param kwargs: Additional parameters for DTW metric
    :return: Silhouette index
    """
    # n_clusters is the number of clusters
    n_clusters = model.n_clusters
    # cluster_k is a list of indices of data points in cluster k
    cluster_k = [X[labels == k] for k in range(n_clusters)]
    s_i = []
    for i in range(X.shape[0]):
        if metric == 'euclidean':
            b = np.min([np.mean(pairwise_distances(X[i].squeeze(-1).unsqueeze(0), cluster_k[j].squeeze(-1))) for j in range(n_clusters) if labels[i] != j])
            a = np.sum(pairwise_distances(X[i].unsqueeze(0), cluster_k[labels[i]])) * (1 / (cluster_k[labels[i]].shape[0] - 1))
        elif metric == 'dtw':
            b = np.min([np.mean(dtw_pairwise_distance(X[i].unsqueeze(0).numpy(), cluster_k[j].numpy(),
                                                      window=kwargs['metric_params']['sakoe_chiba_radius'])) for j in range(n_clusters) if labels[i] != j])
            a = np.mean(dtw_pairwise_distance(X[i].unsqueeze(0).numpy(), cluster_k[labels[i]].numpy(),  window=kwargs['metric_params']['sakoe_chiba_radius']))
        elif metric == 'cross-correlation':
            b = np.min([np.mean(pairwise_cross_correlation(X[i].unsqueeze(0), cluster_k[j])) for j in range(n_clusters) if labels[i] != j])
            a = np.mean(pairwise_cross_correlation(X[i].unsqueeze(0), cluster_k[labels[i]]))
        s_i.append((b-a)/max(b,a))

    return np.mean(s_i)

def dunn_index(X, model, labels, metric, **kwargs):
    """
    Compute the Dunn index of a time series clustering, i.e. the ratio of the minimum inter-cluster distance to the maximum intra-cluster distance
    The lower the better
    :param X: Torch tensor of shape [N, d], N = number of time series, d = number of samples for each
    :param labels: Numpy array of the cluster number for each data point
    :param metric: The similarity measure (euclidean, dtw or cross-correlation)
    :param kwargs: Additional constraint parameters for DTW metric (optional)
    :return: Dunn index
    """
    # n_clusters is the number of clusters
    n_clusters = model.n_clusters
    # cluster_k is a list of indices of data points in cluster k
    cluster_k = [X[labels == k] for k in range(n_clusters)]
    # permutation indices
    joint_indices = [(i,j) for i in range(n_clusters) for j in range(i) if i != j]
    deltas, Deltas, dist = [], [], 0
    # compute min inter-cluster distance
    for (i, j) in joint_indices:
        if metric == 'euclidean':
            dist = pairwise_distances(cluster_k[i], cluster_k[j])
        elif metric == 'dtw':
            dist = dtw_pairwise_distance(cluster_k[i].numpy(), cluster_k[j].numpy(), window=kwargs['metric_params']['sakoe_chiba_radius'])
        elif metric == 'cross-correlation':
            dist = pairwise_cross_correlation(cluster_k[i], cluster_k[j])
        deltas.append(np.min(dist[np.nonzero(dist)]))
    numerator = np.min(deltas)
    # compute max intra-cluster distance
    for k in range(n_clusters):
        if metric == 'euclidean':
            dist = pairwise_distances(cluster_k[k])
        elif metric == 'dtw':
            dist = dtw_pairwise_distance(cluster_k[k].numpy())
        elif metric == 'cross-correlation':
            dist = pairwise_cross_correlation(cluster_k[k], cluster_k[k])
        Deltas.append(np.max(dist))
    denominator = np.max(Deltas)
    DI = numerator/denominator
    return DI


def davies_bouldin_index(X, model, labels, metric, **kwargs):
    """
    Compute the Davies-Bouldin index for a time series clustering. The lower the better.
    :param X: Torch tensor of shape [N, d], N = number of time series, d = number of samples for each
    :param labels: Numpy array of the cluster number for each data point
    :param metric: The similarity measure (euclidean, dtw or cross-correlation)
    :param kwargs: Additional constraint parameters for DTW metric (optional)
    :return: Davies-Bouldin index
    """
    # n_clusters is the number of clusters
    n_clusters = model.n_clusters
    # cluster_k is a list of indices of data points in cluster k
    cluster_k = [X[labels == k] for k in range(n_clusters)]
    # permutation indices
    joint_indices = [(i, j) for i in range(n_clusters) for j in range(i) if i != j]
    ratios, sumdis = [], 0
    # get centroids
    centroids = [np.expand_dims(model.cluster_centers_[k].squeeze(-1), axis=0) for k in range(n_clusters)]
    # compute distances
    Delta_i, Delta_j, delta_ij = 0, 0, 0
    for k in range(n_clusters):
        for (i, j) in joint_indices:
            if metric == 'euclidean':
                Delta_i = np.mean(pairwise_distances(cluster_k[i], centroids[i]))
                Delta_j = np.mean(pairwise_distances(cluster_k[j], centroids[j]))
                delta_ij = euclidean_distances(centroids[i], centroids[j])
            elif metric == 'dtw':
                Delta_i = np.mean(dtw_pairwise_distance(cluster_k[i].numpy(), centroids[i], window=kwargs['metric_params']['sakoe_chiba_radius']))
                Delta_j = np.mean(dtw_pairwise_distance(cluster_k[j].numpy(), centroids[j], window=kwargs['metric_params']['sakoe_chiba_radius']))
                delta_ij = dtw_pairwise_distance(centroids[i], centroids[j], window=kwargs['metric_params']['sakoe_chiba_radius'])
            elif metric == 'cross-correlation':
                Delta_i = np.mean(pairwise_cross_correlation(cluster_k[i], centroids[i]))
                Delta_j = np.mean(pairwise_cross_correlation(cluster_k[j], centroids[j]))
                delta_ij = pairwise_cross_correlation(centroids[i], centroids[j])
            ratios.append((Delta_i + Delta_j)/delta_ij)
        sumdis += np.max(ratios)
    DB = sumdis/n_clusters
    return DB


def calinski_harabasz_index(X, model, labels, metric, **kwargs):
    """
    Compute the Calinski-Harabasz index for time series clustering, i.e. a ratio of between-cluster dispersion (B)
    and within-clsuter dispersion (W). The lower the value, the better the clustering.
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
    factor = (X.shape[1] - n_clusters)/(n_clusters-1)
    # between cluster dispersion
    B = 0
    # centroids is an array of the cluster centroids
    centroids = [model.cluster_centers_.squeeze(-1)[k] for k in range(n_clusters)]
    if metric == 'euclidean':
        for i in range(n_clusters):
            n_i = cluster_k[i].shape[0]
            mu = X.mean(dim=0)
            mu_i = centroids[i]
            B += n_i * ((mu_i - mu.numpy()) ** 2).sum().item()
    elif metric == 'dtw':
        for i in range(n_clusters):
            n_i = cluster_k[i].shape[0]
            mu = elastic_barycenter_average(X.numpy(), metric).squeeze(0)
            mu_i = centroids[i]
            B += n_i * dtw_distance(mu_i, mu, window=kwargs['metric_params']['sakoe_chiba_radius']) ** 2
    elif metric == 'cross-correlation':
        for i in range(n_clusters):
            n_i = cluster_k[i].shape[0]
            mu = cross_correlation_average(X.unsqueeze(-1).numpy()).squeeze(0)
            mu_i = centroids[i]
            B += n_i * distance_cross_correlation(mu_i, mu) ** 2

    # within-cluster dispersion
    W = compute_WCSS(n_clusters=n_clusters, cluster_k=cluster_k, centroids=centroids, metric=metric, **kwargs)
    return (B/W)*factor


def stability_index(X, model, labels, metric, **kwargs):
    """
    Compute stability measures for time series clustering by deleting a percentage of the dataset columns and comparing the
    perturbed clustering with the unperturbed one. The lower, the better.
    :param X:
    :param model:
    :param labels:
    :param perc_col_del:
    :param method:
    :param metric:
    :param kwargs:
    :return:

    Reference: Alboukadel Kassambara (2017). "Practical Guide To Cluster Analysis in R".
    """
    # Number of clusters
    n_clusters = len(np.unique(labels))
    # Get centroids of the clusters
    centroids = [model.cluster_centers_.squeeze(-1)[k] for k in range(n_clusters)]
    # Construct the perturbed dataset as original dataset with a percentage of datapoints removed
    n_col = X.shape[1]
    n_col_removed = int(n_col*kwargs['stability_params']['perc_col_del'])
    n_col_keep = n_col - n_col_removed
    random.seed(42)  # Set a seed for reproducibility
    idx_col_keep = np.sort(random.sample(range(X.shape[1]), n_col_keep))
    X_data_cut = X[:, idx_col_keep].clone()
    labels_cut = model.fit_predict(X_data_cut)

    # Compute Average Proportion of Non-overlap (APN) stability measure
    # Correspond to the proportion observations not placed in the same cluster after perturbation (the lower the better)
    if kwargs['stability_params']['method'] == 'apn':
        apn, total_oij = 0, 0
        for i in range(n_clusters):
            oij_sq = 0
            for j in range(n_clusters):
                total_oij += len(np.flatnonzero(labels_cut[np.flatnonzero(labels==i)]==j))
                oij_sq += (len(np.flatnonzero(labels_cut[np.flatnonzero(labels==i)]==j)))**2
            apn += oij_sq/len(np.flatnonzero(labels==i))
        return 1 - (apn/total_oij)
    # Compute Average Distance (AD) stability measure
    # Correspond to the average distance between observations placed in the same cluster under both cases (the lower the better)
    elif kwargs['stability_params']['method'] == 'ad':
        ad = 0
        for i in range(n_clusters):
            for j in range(n_clusters):
                oij = len(np.flatnonzero(labels_cut[np.flatnonzero(labels==i)]==j))
                if metric == 'euclidean':
                    dij = np.mean(pairwise_distances(X[np.flatnonzero(labels==i)].squeeze(-1), X[np.flatnonzero(labels_cut==j)].squeeze(-1)))
                elif metric == 'dtw':
                    dij = np.mean(dtw_pairwise_distance(X[np.flatnonzero(labels == i)].squeeze(-1), X[np.flatnonzero(labels_cut == j)].squeeze(-1), window=kwargs['metric_params']['sakoe_chiba_radius']))
                elif metric == 'cross-correlation':
                    dij = np.mean(pairwise_cross_correlation(X[np.flatnonzero(labels == i)].squeeze(-1), X[np.flatnonzero(labels_cut == j)].squeeze(-1)))
                ad += oij*dij
        return ad/X.shape[0]
    # Compute the Average Distance between Means (ADM)
    # Correspond to average distance between cluster centroids for observations placed in the same cluster under both cases (the lower the better)
    elif kwargs['stability_params']['method'] == 'adm':
        centroids_cut = [model.cluster_centers_.squeeze(-1)[k] for k in range(n_clusters)]
        adm = 0
        for i in range(n_clusters):
            for j in range(n_clusters):
                oij = len(np.flatnonzero(labels_cut[np.flatnonzero(labels==i)]==j))
                if metric == 'euclidean':
                    dij = euclidean_distances(centroids[i], centroids_cut[j])
                elif metric == 'dtw':
                    dij = dtw_distance(centroids[i], centroids_cut[j], window=kwargs['metric_params']['sakoe_chiba_radius'])
                elif metric == 'cross-correlation':
                    dij = distance_cross_correlation(centroids[i], centroids_cut[j])
                adm += oij*dij
        return adm/X.shape[0]


def hartigan_index(X, model_k, labels_k, model_k1, labels_k1, metric, **kwargs):
    """
    Compute the Hartigan validity index for time series clustering, i.e. the ratio of within-cluster sum of squares for
    cluster k and for cluster k + 1
    The lower the better
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
    return (WCSS1/WCSS2 - 1)/factor


def krzanoswki_lai_index(X, model1, labels1, model2, labels2, model3, labels3, metric, **kwargs):
    """
    Compute the Krzanoswki-Lai validity index for time series clustering
    The lower the better
    """
    # compute WCSS1 for configuration with k-1 clusters
    n_clusters1 = model1.n_clusters
    # cluster_k is a list of indices of data points in cluster k
    cluster_k1 = [X[labels1 == k] for k in range(n_clusters1)]
    # Get centroids
    centroids1 = [model1.cluster_centers_.squeeze(-1)[k] for k in range(n_clusters1)]
    # Compute WCSS for model with k clusters
    WCSS1 = 0
    for i in range(n_clusters1):
        variance = 0
        for datapoint in cluster_k1[i]:
            if metric == 'euclidean':
                variance += ((datapoint - centroids1[i])**2).sum().item()
            elif metric == 'dtw':
                variance += dtw_distance(datapoint.numpy(), centroids1[i], window=kwargs['metric_params']['sakoe_chiba_radius']) **2
            elif metric == 'cross-correlation':
                variance += distance_cross_correlation(datapoint.numpy(), centroids1[i])
        WCSS1 += variance
    # Number of clusters for clustering 2
    n_clusters2 = len(np.unique(labels2))
    # cluster_k2 is a list of indices of data points for k+1 clusters
    cluster_k2 = [X[labels2 == k] for k in range(n_clusters2)]
    centroids2 = [model2.cluster_centers_.squeeze(-1)[k] for k in range(n_clusters2)]
    # Compute WCSS for model with k+1 clusters
    WCSS2 = 0
    for i in range(n_clusters2):
        variance = 0
        for datapoint in cluster_k2[i]:
            if metric == 'euclidean':
                variance += ((datapoint - centroids2[i])**2).sum().item()
            elif metric == 'dtw':
                variance += dtw_distance(datapoint.numpy(), centroids2[i], window=kwargs['metric_params']['sakoe_chiba_radius']) **2
            elif metric == 'cross-correlation':
                variance += distance_cross_correlation(datapoint.numpy(), centroids2[i])
        WCSS2 += variance
    # compute WCSS3 for configuration with k+1 clusters
    n_clusters3 = len(np.unique(labels3))
    # cluster_k is a list of indices of data points in cluster k
    cluster_k3 = [X[labels3 == k] for k in range(n_clusters3)]
    centroids3 = [model3.cluster_centers_.squeeze(-1)[k] for k in range(n_clusters3)]
    WCSS3 = 0
    for i in range(n_clusters3):
        variance = 0
        for datapoint in cluster_k3[i]:
            if metric == 'euclidean':
                variance += ((datapoint - centroids3[i])**2).sum().item()
            elif metric == 'dtw':
                variance += dtw_distance(datapoint.numpy(), centroids3[i], window=kwargs['metric_params']['sakoe_chiba_radius']) **2
            elif metric == 'cross-correlation':
                variance += distance_cross_correlation(datapoint.numpy(), centroids3[i])
        WCSS3 += variance
    KL = (WCSS2/WCSS3 - 1)/(WCSS1/WCSS2 - 1)
    return KL


def k_nn_consistency_index(X, labels, n_neighbors, metric, metric_params):
    """
    Compute a connectivity measure for a time series clustering using k-nearest neighbours. The higher the better.
    """
    model = KNeighborsTimeSeries(n_neighbors=n_neighbors, metric=metric, metric_params=metric_params)
    model.fit(X)
    k_nearest_indices = model.kneighbors(X, n_neighbors=n_neighbors)[1]
    conn_index = 0
    for i in range(X.shape[0]):
        for j in range(n_neighbors):
            if labels[k_nearest_indices[i][j]] != labels[i]:
                conn_index += 1/(j+1)

    return conn_index