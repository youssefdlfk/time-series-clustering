import logging
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tslearn.clustering import KShape, TimeSeriesKMeans

from ts_clustering.algo_specs import AlgorithmSpec, Metric

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class TimeSeriesClustering:
    def __init__(self, X, config):
        self.X = X
        self.config = config
        # Parameters for DTW metric
        dtw_params = {'global_constraint': self.config.global_constraint,
                      'sakoe_chiba_radius': self.config.constraint_radius,
                      'itakura_max_slope': self.config.constraint_slope}
        # Algorithms
        self.algorithms: Dict[str, AlgorithmSpec] = {
            spec.metric.value: spec
            for spec in [
                AlgorithmSpec("K‑Means Euclid", Metric.EUCLIDEAN, TimeSeriesKMeans, config, {}),
                AlgorithmSpec("K‑Means DTW", Metric.DTW, TimeSeriesKMeans, config, dtw_params),
                AlgorithmSpec("K‑Shape", Metric.CC, KShape, config, {}),
            ]
        }

    def run_timeseries_clustering(self, algo_str: str, n_clusters: int) -> tuple[TimeSeriesKMeans, np.ndarray]:
        """
        Run clustering with a specified algorithm and number of clusters.
        :param n_clusters: Number of clusters
        :param algo_str: Key string in the algorithm dictionary referring to a specific clustering algorithm
        :return: Clustering model and labels
        """
        logging.info("Running clustering...")
        model = self.algorithms[algo_str].build(n_clusters=n_clusters)
        labels = model.fit_predict(self.X)
        return model, labels

    def plot_timeseries_clustering(self, model, labels: np.ndarray, algo_str: str,
                                   n_clusters: int, df_insight: pd.DataFrame) -> None:
        """
        Apply clustering and plot the timeseries, their centroids and the insight trial %
        :param model: Clustering model
        :param labels: Clustering labels
        :param algo_str: Key string in the algorithm dictionary referring to a specific clustering algorithm
        :param n_clusters: Number of clusters
        :param df_insight: Dataframe indicating insight or not insight for each time series
        """
        logging.info("Plotting clustering results...")

        plt.figure(figsize=(15, 8))
        nb_insights_total = len(df_insight[df_insight['Insight']==1])
        for cluster in range(n_clusters):
            nb_insights = 0
            indices = np.flatnonzero(labels==cluster)
            plt.subplot(1, n_clusters, cluster+1)
            for idx in indices:
                plt.plot(self.X[idx], "k-", alpha=0.2)
                if df_insight['Insight'].iloc[idx] == 1:
                    nb_insights += 1
            if isinstance(model, KShape):
                # Add back the mean (removed due to z-normalization) to shift the centroid up for better visualization
                centroid = model.cluster_centers_[cluster].flatten() + np.mean(self.X[np.flatnonzero(labels==cluster)])
            else:
                centroid = model.cluster_centers_[cluster].flatten()
            plt.plot(centroid, "r-")
            plt.title(f'Cluster {cluster+1} \n #trials: {len(indices)} \n Insight trials (% of cluster): '
                      f'{(nb_insights/len(indices))*100:.1f}% \n Insight trials (% of total): {(nb_insights/nb_insights_total)*100:.1f}%')
        plt.savefig(f'{algo_str}_{n_clusters}clusters.png')
        plt.show()
