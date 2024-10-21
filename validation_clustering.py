import logging
import numpy as np
from sklearn.preprocessing import minmax_scale
from config import ClusteringConfig
from utils_clustering import spearman_footrule_distance
from timeseries_clustering import TimeSeriesClustering

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ValidationTimeSeriesClustering(TimeSeriesClustering):
    def __init__(self, X: np.ndarray, config: ClusteringConfig):
        super().__init__(X, config)
        if config.k1 >= config.k2:
            raise ValueError(f'Invalid range for number of clusters: k1 must be smaller than k2')
        if config.k1 <= 0 or config.k2 <= 0:
            raise ValueError(f'Invalid range for number of clusters: k1 and k2 have to be positive integers')
        self.k1 = config.k1
        self.k2 = config.k2
        self.nb_algos = len(self.algos_dict)
        self.nb_val_idx = len(self.val_idx_dict)

    def compute_score_matrix(self) -> np.ndarray:
        """
        Compute a score matrix which scores each combination of 'algorithm-#clusters' using the validation indices
        :return: Score matrix
        """
        logging.info("Computing score matrix for clustering algorithms and cluster numbers...")

        # Initialize empty array for the score matrix
        score_matrix = np.empty((self.nb_algos*(self.k2+1-self.k1), self.nb_val_idx))

        # Loop over the clustering algorithms
        for algo_idx, (_, algo_inf) in enumerate(self.algos_dict.items()):
            if algo_inf['metric'] == 'dtw':
                metric_params = algo_inf['metric_params']
            else:
                metric_params = {}

            # Loop over the range of clusters
            for k in range(self.k1, self.k2+1):
                # Train clustering model with k clusters
                model1 = algo_inf['model'](n_clusters=k)
                # Fit and predict the cluster labels
                labels1 = model1.fit_predict(self.X)

                # Loop over the validation indices
                for val_idx, (val_name, val_inf) in enumerate(self.val_idx_dict.items()):
                    # Specify the parameters for the stability indices (apn and ad)
                    if val_name in ['apn', 'ad']:
                        score = val_inf['func'](X=self.X, model=model1, labels=labels1,
                                                metric=algo_inf['metric'], metric_params=metric_params,
                                                stability_params=val_inf['stability_params'])
                    # Specify a model with k+1 clusters required for the hartigan index
                    elif val_name == 'hartigan':
                        model1_k1 = algo_inf['model'](n_clusters=k+1)
                        val_inf['func'](X=self.X, model_k=model1, labels_k=labels1,
                                        model_k1=model1_k1, labels_k1=model1_k1.fit_predict(self.X),
                                        metric=algo_inf['metric'], metric_params=metric_params)
                    # Score is computed the same for all other indices
                    else:
                        score = val_inf['func'](X=self.X, model=model1, labels=labels1,
                                                metric=algo_inf['metric'], metric_params=metric_params)
                    # For silhouette, we make it negative such that the lower the index, the better the clustering
                    if not val_inf['lower_better']:
                        score *= -1
                    # The score is implemented in the corresponding matrix element
                    score_matrix[(k - self.k1) + algo_idx * (self.k2 + 1 - self.k1), val_idx] = score

        return score_matrix

    def _normalize_score_matrix(self, score_matrix: np.ndarray):
        """
        Scale a score matrix to [0,1] along the validation scores for more accurate comparison.
        :param score_matrix: Score matrix
        :return: Normalized score matrix
        """
        # Normalize scores (scaling to [0, 1])
        for idx in range(self.nb_val_idx):
            score_matrix[:, idx] = minmax_scale(score_matrix[:, idx])

        # Save score matrix
        np.save('score_matrix.npy', score_matrix)

        return score_matrix

    @staticmethod
    def _rank_algo(score_matrix: np.ndarray):
        """
        Computes a matrix of the ranks of algorithms-#clusters combinations from a matrix of their validity scores
        :param score_matrix: Matrix of the validity scores of algo-clus combinations
        :return: The rank matrix
        """
        # Give ranking to algorithms-#clsuters for each validation index (0=best score)
        indices = np.argsort(score_matrix, axis=0)
        rank_matrix = np.argsort(indices, axis=0)
        # Save rank matrix
        np.save('rank_matrix.npy', rank_matrix)
        return rank_matrix

    def get_best_algo(self) -> tuple[str, int]:
        """
        Choose the best algorithm-#clusters combination as the closest to a reference ranking using the Spearman
        footrule distance
        :return: A string of the winning algorithm-#clusters combination
        """

        logging.info("Selecting best algorithm and #clusters based on the score matrix...")

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
        for algo_idx, (algo_str, _) in enumerate(self.algos_dict.items()):
            for k in range(self.k1, self.k2):
                algos_clus_dict[(k-self.k1)+algo_idx*(self.k2+1-self.k1)] = (algo_str, k)
        return algos_clus_dict[best_algo_idx]

    @staticmethod
    def save_output_to_file(filename: str, optim_algo: str, optim_n_clusters: int):
        """
        Save the output of the clustering validation in .txt file.
        :param filename: Name of the saved text file
        :param optim_algo: Optimal algorithm name
        :param optim_n_clusters: Optimal number of clusters within the tested range
        """
        with open(filename, "w") as text_file:
            text_file.write(f'Result of Validation: Algorithm: {optim_algo}, N° clusters: {optim_n_clusters}')


