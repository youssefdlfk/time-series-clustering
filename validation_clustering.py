import numpy as np
from sklearn.preprocessing import minmax_scale
from tslearn.clustering import TimeSeriesKMeans, KShape
from validation_indices import (silhouette_index, dunn_index, davies_bouldin_index, calinski_harabasz_index,
                                stability_index, hartigan_index)
from utils_clustering import spearman_footrule_distance


class ValidationTimeSeriesClustering:
    def __init__(self, X, k1, k2, algos_dict, val_idx_dict):
        self.X = X
        self.k1 = k1
        self.k2 = k2
        self.algos_dict = algos_dict
        self.nb_algos = len(algos_dict)
        self.val_idx_dict = val_idx_dict
        self.nb_val_idx = len(val_idx_dict)

    def initialize_score_matrix(self):
        """
        Initialize an empty score matrix with approximate dimensions
        :return: Score matrix
        """
        return np.empty((self.nb_algos*(self.k2-self.k1), self.nb_val_idx))

    def compute_score_matrix(self):
        """
        Compute a score matrix of nb_val_idx validity indices for nb_algos clustering algorithms of a dataset X of time series
        :param X: Tensor dataset of time series of shape [n, d]
        :param k1: Starting number of clusters
        :param k2: Final number of clusters
        :return: Score matrix
        """
        # Initialize empty array for the score matrix
        score_matrix = self.initialize_score_matrix()
        # All validity indices are better when lower, except for silhouette and k_nn_consistency
        # their scores are taken negative such that lower values always mean better clustering

        # Algorithm 1 : K-Means with Euclidean distance
        for k in range(self.k1, self.k2):
            model1 = TimeSeriesKMeans(n_clusters=k, init='k-means++', metric='euclidean', max_iter=10, random_state=42, n_jobs=-1)
            labels1 = model1.fit_predict(self.X)
            score_matrix[(k-self.k1)+0*(self.k2-self.k1), 0] = silhouette_index(self.X, model=model1, labels=labels1, metric='euclidean')*(-1)
            score_matrix[(k-self.k1)+0*(self.k2-self.k1), 1] = dunn_index(self.X, model=model1, labels=labels1, metric='euclidean')
            score_matrix[(k-self.k1)+0*(self.k2-self.k1), 2] = davies_bouldin_index(self.X, model=model1, labels=labels1, metric='euclidean')
            score_matrix[(k-self.k1)+0*(self.k2-self.k1), 3] = calinski_harabasz_index(self.X, model=model1, labels=labels1, metric='euclidean')
            score_matrix[(k-self.k1)+0*(self.k2-self.k1), 4] = stability_index(self.X, model=model1, labels=labels1, method='apn', perc_col_removed=0.1, metric='euclidean')
            score_matrix[(k-self.k1)+0*(self.k2-self.k1), 5] = stability_index(self.X, model=model1, labels=labels1, method='ad', perc_col_removed=0.1, metric='euclidean')
            model1_k1 = TimeSeriesKMeans(n_clusters=k+1, init='k-means++', metric='euclidean', max_iter=10, random_state=42, n_jobs=-1)
            score_matrix[(k-self.k1)+0*(self.k2-self.k1), 6] = hartigan_index(self.X, model_k=model1, labels_k=labels1, model_k1=model1_k1, labels_k1=model1_k1.fit_predict(self.X), metric='euclidean')
            #score_matrix[(k-k1)+0*(self.self.k2-k1), 6] = k_nn_consistency_index(self.X, labels=labels1, n_neighbors=3, metric='euclidean', metric_params={})*(-1)

        # Algorithm 2 : K-Means with DTW distance
        for k in range(self.k1, self.k2):
            metric_params = {'global_constraint': "sakoe_chiba", 'sakoe_chiba_radius': 0.1}
            model2 = TimeSeriesKMeans(n_clusters=k, init='k-means++', metric='dtw', max_iter=1, random_state=42,
                                 n_jobs=-1, metric_params=metric_params)
            labels2 = model2.fit_predict(self.X)
            score_matrix[(k-self.k1)+1*(self.k2-self.k1), 0] = silhouette_index(self.X, model=model2, labels=labels2, metric='dtw', metric_params=metric_params) * (-1)
            score_matrix[(k-self.k1)+1*(self.k2-self.k1), 1] = dunn_index(self.X, model=model2, labels=labels2, metric='dtw', metric_params=metric_params)
            score_matrix[(k-self.k1)+1*(self.k2-self.k1), 2] = davies_bouldin_index(self.X, model=model2, labels=labels2, metric='dtw', metric_params=metric_params)
            score_matrix[(k-self.k1)+1*(self.k2-self.k1), 3] = calinski_harabasz_index(self.X, model=model2, labels=labels2, metric='dtw', metric_params=metric_params)
            score_matrix[(k-self.k1)+1*(self.k2-self.k1), 4] = stability_index(self.X, model=model2, labels=labels2, method='apn', perc_col_removed=0.1, metric='dtw', metric_params=metric_params)
            score_matrix[(k-self.k1)+1*(self.k2-self.k1), 5] = stability_index(self.X, model=model2, labels=labels2, method='ad', perc_col_removed=0.1, metric='dtw', metric_params=metric_params)
            model2_k1 = TimeSeriesKMeans(n_clusters=k+1, init='k-means++', metric='dtw', max_iter=10, random_state=42, n_jobs=-1, metric_params=metric_params)
            score_matrix[(k-self.k1)+1*(self.k2-self.k1), 6] = hartigan_index(self.X, model_k=model2, labels_k=labels2, model_k1=model2_k1, labels_k1=model2_k1.fit_predict(self.X), metric='dtw', metric_params=metric_params)
            #score_matrix[(k-k1)+1*(self.self.k2-k1), 7] = k_nn_consistency_index(self.X, labels=labels2, n_neighbors=3, metric='dtw', metric_params=metric_params) * (-1)

        # Algorithm 3 : K-Shape with cross-correlation distance
        for k in range(self.k1, self.k2):
            model3 = KShape(n_clusters=k, max_iter=10, random_state=42)
            labels3 = model3.fit_predict(self.X)
            score_matrix[(k-self.k1)+2*(self.k2-self.k1), 0] = silhouette_index(self.X, model=model3, labels=labels3, metric='cross-correlation') * (-1)
            score_matrix[(k-self.k1)+2*(self.k2-self.k1), 1] = dunn_index(self.X, model=model3, labels=labels3, metric='cross-correlation')
            score_matrix[(k-self.k1)+2*(self.k2-self.k1), 2] = davies_bouldin_index(self.X, model=model3, labels=labels3, metric='cross-correlation')
            score_matrix[(k-self.k1)+2*(self.k2-self.k1), 3] = calinski_harabasz_index(self.X, model=model3, labels=labels3, metric='cross-correlation')
            score_matrix[(k-self.k1)+2*(self.k2-self.k1), 4] = stability_index(self.X, model=model3, labels=labels3, method='apn', perc_col_removed=0.1, metric='cross-correlation')
            score_matrix[(k-self.k1)+2*(self.k2-self.k1), 5] = stability_index(self.X, model=model3, labels=labels3, method='ad', perc_col_removed=0.1, metric='cross-correlation')
            model3_k1 = KShape(n_clusters=k+1, max_iter=10, random_state=42)
            score_matrix[(k-self.k1)+2*(self.k2-self.k1), 6] = hartigan_index(self.X, model_k=model3, labels_k=labels3, model_k1=model3_k1, labels_k1=model3_k1.fit_predict(self.X), metric='cross-correlation')
            #score_matrix[(k-k1)+2*(self.k2-k1), 7] = k_nn_consistency_index(self.X, labels=labels3, n_neighbors=3, metric='cross-correlation', metric_params={}) * (-1)

        return score_matrix

    def normalize_score_matrix(self, score_matrix):
        """
        Scale a score matrix to [0,1] along the validation scores for more accurate comparison.
        :param score_matrix: Score matrix
        :return: Normalized score matrix
        """
        # Normalize scores (scaling to [0, 1])
        for idx in range(self.nb_val_idx):
            score_matrix[:, idx] = minmax_scale(score_matrix[:, idx])

        return score_matrix

    @staticmethod
    def rank_algo(score_matrix):
        """
        Computes a matrix of the ranks of algorithms-#clusters combinations from a matrix of their validity scores
        :param score_matrix: Matrix of the validity scores of algo-clus combinations
        :return: The rank matrix
        """
        indices = np.argsort(score_matrix, axis=0)
        rank_matrix = np.argsort(indices, axis=0)
        return rank_matrix

    def get_best_algo(self):
        """
        Choose the best algorithm-#clusters combination using the Spearman footrule distance on their ranks
        :return: A string of the winning algorithm-#clusters combination
        """
        # Compute the score matrix
        score_matrix = self.compute_score_matrix()
        # Normalize score matrix
        score_matrix = self.normalize_score_matrix(score_matrix)
        # Compute the rank matrix
        rank_matrix = self.rank_algo(score_matrix)

        # Each rank vector of algo-clus combination is compared to an ideal reference ranking filled with zeros
        ranking_ref = np.zeros(self.nb_val_idx)
        dist_to_ref = []

        # The distance to the reference is computed using the spearman footrule distance
        for algo_clus_idx in range(score_matrix.shape[0]):
            dist_to_ref.append(spearman_footrule_distance(ranking_ref, rank_matrix[algo_clus_idx, :]))

        # The best algo-clus combination is the one closest to the reference ranking vector
        best_algo_idx = dist_to_ref.index(min(dist_to_ref))

        # Get the corresponding algorithm name and number of clusters
        algos_clus_dict = {}
        for i in self.algos_dict.keys():
            for k in range(self.k1, self.k2):
                algos_clus_dict[(k-self.k1)+i*(self.k2-self.k1)] = (self.algos_dict[i], k)
        return algos_clus_dict[best_algo_idx]


