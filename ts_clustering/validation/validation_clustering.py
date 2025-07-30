"""
Module for validating clustering algorithms applied to time series data.

Extends TimeSeriesClustering to include functionality for evaluating clusters using multiple validity indices,
ranking algorithm-cluster number combinations, and selecting optimal clustering configurations based on
comprehensive metric evaluations.
"""

import collections
import copy
import logging
import pickle
import random
from typing import List

import numpy as np

from config import ClusteringConfig
from ts_clustering.clustering.timeseries_clustering import TimeSeriesClustering
from ts_clustering.clustering.utils import (compute_distance_matrix,
                                            spearman_footrule_distance)
from ts_clustering.validation.index_specs import (ADIndex, APNIndex,
                                                  CalinskiHarabaszIndex,
                                                  DaviesBouldinIndex,
                                                  DunnIndex, HartiganIndex,
                                                  SilhouetteIndex,
                                                  ValidityIndex)

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
        self.validation_indices: List[ValidityIndex] = [
            SilhouetteIndex(),
            DunnIndex(),
            DaviesBouldinIndex(),
            CalinskiHarabaszIndex(),
            APNIndex(perc_col_del=self.config.perc_col_del),
            ADIndex(perc_col_del=self.config.perc_col_del),
            HartiganIndex(),
        ]
    def compute_score_matrix(self) -> np.ndarray:
        """
        Compute a score matrix which scores each combination of 'algorithm-#clusters' using the validation indices.
        One dimension contains algorithm-#clusters combinations in this order.
        (e.g. euclidean-2clusters, euclidean-3clusters, dtw-2clusters, dtw-3clusters, kshape-2clusters,
        kshape-3clusters).
        :return: Score matrix
        """
        logging.info("Computing score matrix for clustering algorithms and cluster numbers...")
        # Initialize empty array for the score matrix
        nb_algos = len(self.algorithms)
        nb_val_idx = len(self.validation_indices)
        n_k = self.k2 - self.k1 + 1
        score_matrix = np.empty((nb_algos*n_k, nb_val_idx), dtype=float)
        # Initialize dictionary of models and labels for all algos and #clusters
        models_labels_dict = collections.defaultdict(dict)
        row_offset = 0
        # Loop over the clustering algorithms
        for algo_spec in self.algorithms.values():
            metric = algo_spec.metric.value
            metric_params = algo_spec.metric_params

            logging.info(f"Computing distance matrix for {metric}...")
            # 1) Compute distance matrix
            distance_matrix = compute_distance_matrix(X=self.X, metric=metric, metric_params=metric_params)
            # save matrix
            logging.info(f"Saving distance matrix for {metric}...")
            np.save('dist_mat_'+metric, distance_matrix)

            # 2) Fit models
            # model and labels dictionary
            mod_lab_dict = {}
            lab_cut_dict = {}
            # Loop over the range of clusters (and one more for the Hartigan index)
            for k in range(self.k1, self.k2 + 2):
                logging.info(f"Fit and predict model for cluster {k}...")
                mod_lab_dict, lab_cut_dict = self._fit_models(algo_spec, metric, mod_lab_dict, lab_cut_dict,
                                                              models_labels_dict, k)

            # 3) Compute validation indices
            # Loop over the range of clusters
            for k in range(self.k1, self.k2 + 1):
                logging.info(f"Computing validation indices for cluster {k}...")
                score_matrix = self._compute_indices(metric, score_matrix, mod_lab_dict, lab_cut_dict, distance_matrix,
                                                     metric_params, k, row_offset)
            # Save score matrix
            self._save_results(score_matrix, models_labels_dict)

        return score_matrix

    def _fit_models(self, algo_spec, metric_str, mod_lab_dict, lab_cut_dict, models_labels_dict, k):
        model_k = algo_spec.build(n_clusters=k)
        labels_k = model_k.fit_predict(self.X)
        mod_lab_dict[k] = (model_k, labels_k)
        models_labels_dict[metric_str][k] = (model_k, labels_k)
        # Perturbed labels only for k ≤ k2 when stability indices are used
        if k <= self.k2:
            logging.info(f"[{metric_str}]     • computing perturbed labels for stability...")
            labels_k_cut = self._get_perturbed_labels(
                X=self.X,
                model=model_k,
                stability_params={"perc_col_del": self.config.perc_col_del}
            )
            lab_cut_dict[k] = labels_k_cut
        # Saving progress
        logging.info("Saving models and labels progress...")
        with open(f'models_labels_dict.pkl', 'wb') as file:
            pickle.dump(models_labels_dict, file)
        logging.info("Dictionary saved!")

        return mod_lab_dict, lab_cut_dict

    def _compute_indices(self, metric_str, score_matrix, mod_lab_dict, lab_cut_dict, distance_matrix, metric_params, k,
                         row_offset):
        model_k, labels_k = mod_lab_dict[k]
        # Build context passed to each index
        context = {
            "X":               self.X,
            "labels":          labels_k,
            "distance_matrix": distance_matrix,
            "model":           model_k,
            "metric":          metric_str,
            "metric_params":   metric_params
        }
        # Hartigan needs k+1
        if "hartigan" in {idx.name for idx in self.validation_indices}:
            model_k1, labels_k1 = mod_lab_dict[k + 1]
            context.update({"model_k1": model_k1, "labels_k1": labels_k1})

        # Stability indices need the perturbed labels
        if k in lab_cut_dict:
            context["labels_cut"] = lab_cut_dict[k]

        # Compute each index
        for idx_col, idx_obj in enumerate(self.validation_indices):
            logging.info(f"[{metric_str}]   k={k} → computing '{idx_obj.name}'...")
            try:
                score = idx_obj.compute(**context)
            except Exception as e:
                logging.exception(
                    f"[{metric_str}]   k={k} index='{idx_obj.name}' failed; defaulting to NaN"
                )
                score = np.nan

            # Sign so that “lower is better” always
            if not idx_obj.lower_better:
                score = -score

            # Store in the correct row/column
            row = row_offset + (k - self.k1)
            score_matrix[row, idx_col] = score

        return score_matrix

    @staticmethod
    def _save_results(score_matrix, models_labels_dict):
        logging.info("Saving intermediate models & score matrix...")
        np.save('score_matrix.npy', score_matrix)
        with open('models_labels_dict.pkl', 'wb') as file:
            pickle.dump(models_labels_dict, file)
        logging.info("Score matrix and dictionary saved!")


    @staticmethod
    def _get_perturbed_labels(X, model, **kwargs):
        # Construct the perturbed dataset as original dataset with a percentage of datapoints removed
        n_col = X.shape[1]
        n_col_removed = int(n_col * kwargs['stability_params']['perc_col_del'])
        n_col_keep = n_col - n_col_removed
        random.seed(42)  # Set a seed for reproducibility
        idx_col_keep = np.sort(random.sample(range(X.shape[1]), n_col_keep))
        X_data_cut = copy.deepcopy(X[:, idx_col_keep])
        model_cut = copy.deepcopy(model)
        labels_cut = model_cut.fit_predict(X_data_cut)
        return labels_cut

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
        return rank_matrix

    def get_best_algo(self) -> tuple[str, int]:
        """
        Choose the best algorithm-#clusters combination as the closest to a reference ranking using the Spearman
        footrule distance
        :return: A string of the winning algorithm-#clusters combination
        """
        # Compute the score matrix
        score_matrix = self.compute_score_matrix()
        # Save score matrix
        np.save('score_matrix.npy', score_matrix)
        # Compute the rank matrix
        rank_matrix = self._rank_algo(score_matrix)
        # Save rank matrix
        np.save('rank_matrix.npy', rank_matrix)
        # Each rank vector of algo-clus combination is compared to an ideal reference ranking filled with zeros
        ranking_ref = np.zeros(len(self.validation_indices))
        dist_to_ref = []
        # The distance to the reference is computed using the spearman footrule distance
        for algo_clus_idx in range(score_matrix.shape[0]):
            dist_to_ref.append(spearman_footrule_distance(ranking_ref, rank_matrix[algo_clus_idx, :]))
        # The best algo-clus combination is the one closest to the reference ranking vector
        best_algo_idx = dist_to_ref.index(min(dist_to_ref))
        # Get the corresponding algorithm name and number of clusters
        algos_clus_dict = {}
        n_k = self.k2 - self.k1 + 1
        for algo_idx, algo_str in enumerate(self.algorithms.values()):
            for k in range(self.k1, self.k2+1):
                algos_clus_dict[(k-self.k1)+algo_idx*n_k] = (algo_str, k)
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


