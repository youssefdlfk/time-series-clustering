import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tslearn.clustering import KShape, TimeSeriesKMeans

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class TimeSeriesClustering:
    def __init__(self, X, config):
        self.X = X
        self.config = config
        self.algos_dict = self._initialize_algo_mapping()

    def _initialize_algo_mapping(self) -> dict:
        # Parameters for DTW metric
        dtw_params = {'global_constraint': "sakoe_chiba", 'sakoe_chiba_radius': self.config.window_size_perc}
        # Dictionaries of the algorithms and indices used for validation
        algos_dict = {
            'euclidean': {'name': 'K-means', 'metric': 'euclidean',
                          'model': lambda n_clusters: TimeSeriesKMeans(init='k-means++',
                                                                       max_iter=self.config.max_iter,
                                                                       tol=self.config.tol,
                                                                       metric='euclidean',
                                                                       random_state=self.config.random_seed,
                                                                       n_jobs=-1,
                                                                       n_clusters=n_clusters)},
            'dtw': {'name': 'K-means', 'metric': 'dtw', 'metric_params': dtw_params,
                    'model': lambda n_clusters: TimeSeriesKMeans(init='k-means++',
                                                                 max_iter=self.config.max_iter,
                                                                 tol=self.config.tol,
                                                                 random_state=self.config.random_seed,
                                                                 n_jobs=-1, metric='dtw',
                                                                 metric_params=dtw_params,
                                                                 n_clusters=n_clusters)},
            'kshape': {'name': 'K-shape', 'metric': 'cross-correlation',
                       'model': lambda n_clusters: KShape(max_iter=self.config.max_iter,
                                                          tol=self.config.tol,
                                                          random_state=self.config.random_seed,
                                                          n_clusters=n_clusters)
                       }
        }
        return algos_dict

    def run_timeseries_clustering(self, algo_str: str, n_clusters: int) -> tuple[TimeSeriesKMeans, np.ndarray]:
        """
        Run clustering with a specified algorithm and number of clusters.
        :param n_clusters: Number of clusters
        :param algo_str: Key string in the algorithm dictionary referring to a specific clustering algorithm
        :return: Clustering model and labels
        """
        logging.info("Running clustering...")

        model = self.algos_dict[algo_str]['model'](n_clusters=n_clusters)
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
