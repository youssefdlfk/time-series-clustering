import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from data_processing_insight import data_proc_insight
from tslearn.clustering import TimeSeriesKMeans, KShape
from validation_indices import (silhouette_index, dunn_index, davies_bouldin_index, calinski_harabasz_index,
                                stability_index, hartigan_index)


@dataclass
class ClusteringConfig:
    """Configuration parameters for time series clustering"""
    # Length of time series
    ts_length: int
    # Name of the csv data file
    csv_name: str
    # Maximum number of iterations for training of clustering models
    max_iter: int
    # Constraining window (sakoe-chiba) on the DTW matrix as a percentage of the time series length
    window_size_perc: float
    # Percentage of data points removed in the perturbed clusters for stability measures
    perc_col_del: float
    # Range of the number of clusters to explore for validation
    k1: int
    k2: int
    # Down sampling factor of time series to manage memory usage for DTW
    down_sample_factor: int
    # Random seed for reproducibility
    random_seed: int = 42

class TimeSeriesClustering:
    def __init__(self, X, config):
        self.X = X
        self.config = config
        self.algos_dict = self._initialize_algo_mapping()
        self.val_idx_dict = self._initialize_val_idx()

    def _initialize_algo_mapping(self) -> dict:
        # Window size as an integer
        window_size = self.config.window_size_perc
        # Parameters for DTW metric
        dtw_params = {'global_constraint': "sakoe_chiba", 'sakoe_chiba_radius': window_size}
        # Dictionaries of the algorithms and indices used for validation
        algos_dict = {
            'euclidean': {'name': 'K-means', 'metric': 'euclidean',
                          'model': lambda n_clusters: TimeSeriesKMeans(init='k-means++',
                                                                       max_iter=self.config.max_iter,
                                                                       metric='euclidean',
                                                                       random_state=self.config.random_seed,
                                                                       n_clusters=n_clusters)},
            'dtw': {'name': 'K-means', 'metric': 'dtw', 'metric_params': dtw_params,
                    'model': lambda n_clusters: TimeSeriesKMeans(init='k-means++',
                                                                 max_iter=self.config.max_iter,
                                                                 random_state=self.config.random_seed,
                                                                 n_jobs=-1, metric='dtw',
                                                                 metric_params=dtw_params,
                                                                 n_clusters=n_clusters)},
            'kshape': {'name': 'K-shape', 'metric': 'cross-correlation',
                       'model': lambda n_clusters: KShape(max_iter=self.config.max_iter,
                                                          random_state=self.config.random_seed,
                                                          n_clusters=n_clusters)
                       }
        }
        return algos_dict

    def _initialize_val_idx(self) -> dict:
        val_idx_dict = {
            'silhouette': {'func': silhouette_index, 'lower_better': False},
            'dunn': {'func': dunn_index, 'lower_better': True},
            'davies-bouldin': {'func': davies_bouldin_index, 'lower_better': True},
            'calinski-harabasz': {'func': calinski_harabasz_index, 'lower_better': True},
            'apn': {
                'func': stability_index,
                'stability_params': {'method': 'apn', 'perc_col_del': self.config.perc_col_del},
                'lower_better': True},
            'ad': {
                'func': stability_index,
                'stability_params': {'method': 'ad', 'perc_col_del': self.config.perc_col_del},
                'lower_better': True},
            'hartigan': {'func': hartigan_index, 'hartigan_params': {}, 'lower_better': True
                         }
        }

        return val_idx_dict

    def run_timeseries_clustering(self, algo_str: str, n_clusters: int) -> tuple[TimeSeriesKMeans, np.ndarray]:
        """
        Run a clustering algorithm.
        :param n_clusters: Number of clusters
        :param algo_str: Key string in the algorithm dictionary referring to a specific clustering algorithm
        :return: Clustering model and labels
        """
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
                # Add back the mean after z-normalization to shift the centroid up for better visualization
                centroid = model.cluster_centers_[cluster].flatten() + np.mean(self.X[np.flatnonzero(labels==cluster)])
            else:
                centroid = model.cluster_centers_[cluster].flatten()
            plt.plot(centroid, "r-")
            plt.title(f'Cluster {cluster+1} \n #trials: {len(indices)} \n Insight trials (% of cluster): '
                      f'{(nb_insights/len(indices))*100:.1f}% \n Insight trials (% of total): {(nb_insights/nb_insights_total)*100:.1f}%')
        plt.savefig(f'{algo_str}_{n_clusters}clusters.png')
        plt.show()


if __name__ == '__main__':
    config = ClusteringConfig(ts_length=2100,
                              csv_name="2100interpolationfulldata.csv",
                              max_iter=1,
                              window_size_perc=0.05,
                              perc_col_del=0.1,
                              k1=2,
                              k2=3,
                              down_sample_factor=14)

    # Load and process time series data and extract dataframe on insight or not
    X_data, df_insight = data_proc_insight(csv_name=config.csv_name, timeseries_length=config.ts_length,
                                           down_sample_factor=config.down_sample_factor)

    # Instantiation of timeseriesclustering object
    clusterer = TimeSeriesClustering(X=X_data, config=config)

    # Instantiation of validationtimeseriesclustering object
    from validation_clustering import ValidationTimeSeriesClustering
    validator = ValidationTimeSeriesClustering(X=X_data, config=config)

    # Get optimal combination of algorithm and number of clusters
    optim_algo, optim_n_clusters = validator.get_best_algo()

    # Save output of validation
    validator.save_output_to_file('output_validation.txt', optim_algo, optim_n_clusters)

    # Run clustering with optimal algorithm and number of clusters
    model, labels = clusterer.run_timeseries_clustering(optim_algo, optim_n_clusters)

    # Save best model
    with open('optim_model.pkl', 'wb') as file:
        pickle.dump(model, file)

    # Save best cluster labels
    np.save('optim_labels.npy', labels)

    # Plot time series clusters with centroids and % insights
    clusterer.plot_timeseries_clustering(model=model, labels=labels, algo_str=optim_algo, n_clusters=optim_n_clusters,
                                         df_insight=df_insight)
