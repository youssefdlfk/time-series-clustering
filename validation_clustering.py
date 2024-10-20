import numpy as np
from sklearn.preprocessing import minmax_scale
from utils_clustering import spearman_footrule_distance


class ValidationTimeSeriesClustering:
    def __init__(self, X, k1, k2, max_iter, algos_dict, val_idx_dict):
        self.X = X
        self.k1 = k1
        self.k2 = k2
        self.max_iter = max_iter
        self.algos_dict = algos_dict
        self.nb_algos = len(algos_dict)
        self.val_idx_dict = val_idx_dict
        self.nb_val_idx = len(val_idx_dict)

    def _initialize_score_matrix(self):
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
        score_matrix = self._initialize_score_matrix()

        # Loop over the clustering algorithms
        for algo_idx in self.algos_dict:
            if self.algos_dict[algo_idx]['metric'] == 'dtw':
                metric_params = self.algos_dict[algo_idx]['metric_params']
            else:
                metric_params = {}
            # Loop over the range of clusters
            for k in range(self.k1, self.k2+1):
                # Train clustering model with k clusters
                model1 = self.algos_dict[algo_idx]['model'](n_clusters=k)
                # Fit and predict the cluster labels
                labels1 = model1.fit_predict(self.X)
                # Loop over the validation indices
                for val_idx in self.val_idx_dict:
                    # Specify the parameters for the stability indices (apn and ad)
                    if self.val_idx_dict[val_idx]['name'] in ['apn', 'ad']:
                        score = self.val_idx_dict[val_idx]['func'](X=self.X, model=model1, labels=labels1,
                                                                   metric=self.algos_dict[algo_idx]['metric'],
                                                                   metric_params=metric_params,
                                                                   stability_params=self.val_idx_dict[val_idx]['stability_params'])
                    # Specify a model with k+1 clusters required for the hartigan index
                    elif self.val_idx_dict[val_idx]['name'] == 'hartigan':
                        model1_k1 = self.algos_dict[algo_idx]['model'](n_clusters=k+1)
                        score = self.val_idx_dict[val_idx]['func'](X=self.X, model_k=model1, labels_k=labels1,
                                                                   model_k1=model1_k1, labels_k1=model1_k1.fit_predict(self.X),
                                                                   metric=self.algos_dict[algo_idx]['metric'],
                                                                   metric_params=metric_params)
                    # Score is computed the same for all other indices
                    else:
                        score = self.val_idx_dict[val_idx]['func'](X=self.X, model=model1, labels=labels1,
                                                                   metric=self.algos_dict[algo_idx]['metric'],
                                                                   metric_params=metric_params)
                    # For silhouette, we make it negative such that the lower the index, the better the clustering
                    if not self.val_idx_dict[val_idx]['lower_better']:
                        score = score*(-1)

                    # The score is implemented in the corresponding matrix element
                    score_matrix[(k - self.k1) + algo_idx * (self.k2 - self.k1), val_idx] = score

        return score_matrix

    def _normalize_score_matrix(self, score_matrix):
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
    def _rank_algo(score_matrix):
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
        score_matrix = self._normalize_score_matrix(score_matrix)
        # Compute the rank matrix
        rank_matrix = self._rank_algo(score_matrix)

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

    @staticmethod
    def save_output_to_file(filename, optim_algo, optim_n_clusters):
        """
        Save the output of the clustering validation in .txt file.
        :param filename: Name of the saved text file
        :param optim_algo: Optimal algorithm name
        :param optim_n_clusters: Optimal number of clusters within the tested range
        """
        with open(filename, "w") as text_file:
            text_file.write(f'Algorithm: {optim_algo}, N° clusters: {optim_n_clusters}')


