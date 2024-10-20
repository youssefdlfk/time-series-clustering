import numpy as np
import matplotlib.pyplot as plt
from data_processing_insight import data_proc_insight
from tslearn.clustering import TimeSeriesKMeans, KShape
from validation_clustering import ValidationTimeSeriesClustering
from validation_indices import (silhouette_index, dunn_index, davies_bouldin_index, calinski_harabasz_index,
                                stability_index, hartigan_index)

RAND_SEED = 42

class TimeSeriesClustering:
    def __init__(self, X, algos_dict):
        self.X = X
        self.algos_dict = algos_dict

    def run_timeseries_clustering(self, algo_str, n_clusters):
        """
        Run a clustering algorithm.
        :param algo: Algorithm
        :param n_clusters: Number of clusters
        :return:
        """
        model = self.algos_dict[algo_str]['model'](n_clusters=n_clusters)
        Y_data = model.fit_predict(self.X)
        return model, Y_data

    def plot_timeseries_clustering(self, model, labels, algo_str, n_clusters, df_insight):
        """
        Apply clustering and plot the timeseries, their centroids and the insight trial %
        :param X_data: Tensor of the time series data
        :param algo: Clustering algorithm to use (K-means with Euclidean, K-means with DTW, K-Shape with cross-correlation)
        :param n_clusters: Number of clusters
        :param df_insight: Dataframe indicating insight or not insight for each time series
        :param kwargs: Additional parameters for the DTW metric
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
            # /!\The centroids seem shifted down ... we could shift it back up but not sure why it's shifted in the first place
            centroid = model.cluster_centers_[cluster].flatten()
            plt.plot(centroid, "r-")
            plt.title(f'cluster #{cluster+1} \n #trials: {len(indices)} \n insight trials (% of cluster): '
                      f'{(nb_insights/len(indices))*100:.1f}% \n insight trials (% of total): {(nb_insights/nb_insights_total)*100:.1f}%')
        plt.savefig(f'{algo_str}_{n_clusters}clusters.png')
        plt.show()


if __name__ == '__main__':
    # Length of time series
    TS_LENGTH = 2100
    # Name of the csv data file
    CSV_NAME = "2100interpolationfulldata.csv"
    # Maximum number of iterations for training of clustering models
    MAX_ITER = 1
    # Constraining window (sakoe-chiba) on the DTW matrix as a percentage of the time series length
    WINDOW_SIZE_PERC = 0.1
    # Percentage of data points removed in the perturbed clusters for stability measures
    PERC_COL_DEL = 0.1
    # Range of the number of clusters to explore for validation
    K1, K2 = 2, 6
    # Down sampling factor of time series to manage memory usage for DTW
    DOWN_SAMPLE_FACTOR = 6

    # Process time series data and extract dataframe on insight or not
    X_data, df_insight = data_proc_insight(csv_name=CSV_NAME, timeseries_length=TS_LENGTH)
    # Down sampling of the time series data
    X_data = X_data[:, ::DOWN_SAMPLE_FACTOR]
    # Window size as an integer
    window_size = int(WINDOW_SIZE_PERC*X_data.shape[0])
    # Parameters for DTW metric
    dtw_params = {'global_constraint': "sakoe_chiba", 'sakoe_chiba_radius': window_size}

    # Dictionaries of the algorithms and indices used for validation
    algos_dict = {'euclidean': {'name': 'K-means', 'metric': 'euclidean',
                                'model': lambda n_clusters: TimeSeriesKMeans(init='k-means++', max_iter=MAX_ITER,
                                                                             metric='euclidean', random_state=RAND_SEED,
                                                                             n_clusters=n_clusters)},
                  'dtw': {'name': 'K-means', 'metric': 'dtw', 'metric_params': dtw_params,
                          'model': lambda n_clusters: TimeSeriesKMeans(init='k-means++', max_iter=MAX_ITER,
                                                                       random_state=RAND_SEED,
                                                                       n_jobs=-1, metric='dtw',
                                                                       metric_params=dtw_params,
                                                                       n_clusters=n_clusters)},
                  'kshape': {'name': 'K-shape', 'metric': 'cross-correlation',
                             'model': lambda n_clusters: KShape(max_iter=MAX_ITER, random_state=RAND_SEED,
                                                                n_clusters=n_clusters)}}
    val_idx_dict = {'silhouette': {'func': silhouette_index, 'lower_better': False},
                    'dunn': {'func': dunn_index, 'lower_better': True},
                    'davies-bouldin': {'func': davies_bouldin_index, 'lower_better': True},
                    'calinski-harabasz': {'func': calinski_harabasz_index, 'lower_better': True},
                    'apn': {'func': stability_index, 'stability_params': {'method': 'apn', 'perc_col_del': PERC_COL_DEL}, 'lower_better': True},
                    'ad': {'func': stability_index, 'stability_params': {'method': 'ad', 'perc_col_del': PERC_COL_DEL}, 'lower_better': True},
                    'hartigan': {'func': hartigan_index, 'hartigan_params': {}, 'lower_better': True}}

    # Instantiation of timeseriesclustering object
    tsc = TimeSeriesClustering(X=X_data, algos_dict=algos_dict)

    # Instantiation of validationtimeseriesclustering object
    val_tsc = ValidationTimeSeriesClustering(X=X_data, k1=K1, k2=K2, algos_dict=algos_dict, val_idx_dict=val_idx_dict)

    # Get optimal combination of algorithm and number of clusters
    optim_algo, optim_n_clusters = val_tsc.get_best_algo()

    # Save output of validation
    val_tsc.save_output_to_file('output_validation.txt', optim_algo, optim_n_clusters)

    # Run clustering with optimal algorithm and number of clusters
    model, labels = tsc.run_timeseries_clustering(optim_algo, optim_n_clusters)

    # Plot time series clusters with centroids and % insights
    tsc.plot_timeseries_clustering(model=model, labels=labels, algo_str=optim_algo, n_clusters=optim_n_clusters, df_insight=df_insight)
