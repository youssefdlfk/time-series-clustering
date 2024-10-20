import numpy as np
import matplotlib.pyplot as plt
from data_processing_insight import data_proc_insight
from tslearn.clustering import TimeSeriesKMeans, KShape
from validation_clustering import ValidationTimeSeriesClustering
from validation_indices import (silhouette_index, dunn_index, davies_bouldin_index, calinski_harabasz_index,
                                stability_index, hartigan_index)

RAND_SEED = 42

class TimeSeriesClustering:
    def __init__(self, X, max_iter, dtw_params):
        self.X = X
        self.max_iter = max_iter
        self.dtw_params = dtw_params

    # def validate_clustering(self, k1, k2):
    #     """
    #     Find the optimal algorithm and number of clusters according to a set of validation indices and a voting rule.
    #     :param k1: Minimum number of clusters to test.
    #     :param k2: Maximum number of clusters to test.
    #     :return: Optimal algorithm name and optimal number of clusters
    #     """
    #     val_tsc = ValidationTimeSeriesClustering(X=self.X, k1=k1, k2=k2, algos_dict=self.algos_dict, val_idx_dict=self.val_idx_dict)
    #     optim_algo, optim_n_clusters = val_tsc.get_best_algo()
    #     return optim_algo, optim_n_clusters


    def run_timeseries_clustering(self, algo, n_clusters):
        """
        Run a clustering algorithm.
        :param algo: Algorithm
        :param n_clusters: Number of clusters
        :return:
        """
        if algo == 'euclidean':
            model = TimeSeriesKMeans(n_clusters=n_clusters, init='k-means++', max_iter=self.max_iter, metric='euclidean', random_state=RAND_SEED)
        elif algo == 'dtw':
            model = TimeSeriesKMeans(n_clusters=n_clusters, init='k-means++', max_iter=self.max_iter, random_state=RAND_SEED,
                                 n_jobs=-1, metric='dtw', metric_params=self.dtw_params)
        elif algo == 'kshape':
            model = KShape(n_clusters=n_clusters, max_iter=self.max_iter, random_state=RAND_SEED)
        Y_data = model.fit_predict(self.X)

        return model, Y_data

    def plot_timeseries_clustering(self, algo, n_clusters, df_insight):
        """
        Apply clustering and plot the timeseries, their centroids and the insight trial %
        :param X_data: Tensor of the time series data
        :param algo: Clustering algorithm to use (K-means with Euclidean, K-means with DTW, K-Shape with cross-correlation)
        :param n_clusters: Number of clusters
        :param df_insight: Dataframe indicating insight or not insight for each time series
        :param max_iterations: Maximum number of iterations for the clustering algorithms
        :param kwargs: Additional parameters for the DTW metric
        """
        model, Y_data = self.run_timeseries_clustering(algo=algo, n_clusters=n_clusters)
        plt.figure(figsize=(15, 8))
        nb_insights_total = len(df_insight[df_insight['Insight']==1])
        for cluster in range(n_clusters):
            nb_insights = 0
            indices = np.flatnonzero(Y_data== cluster)
            plt.subplot(1, n_clusters, cluster+1)
            for idx in indices:
                plt.plot(self.X[idx], "k-", alpha=0.2)
                if df_insight['Insight'].iloc[idx] == 1:
                    nb_insights += 1
            # /!\The centroids seem shifted down ... we could shift it back up but not sure why it's shifted in the first place
            centroid = model.cluster_centers_[cluster].flatten()
            plt.plot(centroid, "r-")
            plt.title(f'cluster #{cluster+1} \n #trials: {len(indices)} \n insight trials (% of cluster): {(nb_insights/len(indices))*100:.1f}% \n insight trials (% of total): {(nb_insights/nb_insights_total)*100:.1f}%')
        plt.savefig(f'{algo}_{n_clusters}clusters.png')
        plt.show()


if __name__ == '__main__':
    # Loading & processing data
    timeseries_length = 2100
    csv_name = "2100interpolationfulldata.csv"
    X_data, df_insight = data_proc_insight(csv_name=csv_name, timeseries_length=timeseries_length)

    # Constraining parameters to manage memory usage for DTW
    window_size = int(0.1*X_data.shape[0])  # window size (sakoe_chiba)
    dtw_params = {'global_constraint': "sakoe_chiba", 'sakoe_chiba_radius': window_size}
    down_sample_factor = 6
    max_iter = 1

    # Down sampling of the time series data
    X_data = X_data[:, ::down_sample_factor]

    # Dictionaries of the algorithms and indices used for validation
    algos_dict = {0: {'name': 'K-means', 'metric': 'euclidean',
                      'model': lambda n_clusters: TimeSeriesKMeans(init='k-means++', max_iter=max_iter,
                                                                   metric='euclidean', random_state=RAND_SEED,
                                                                   n_clusters=n_clusters)},
                  1: {'name': 'K-means', 'metric': 'dtw', 'metric_params': dtw_params,
                      'model': lambda n_clusters: TimeSeriesKMeans(init='k-means++', max_iter=max_iter,
                                                                   random_state=RAND_SEED,
                                                                   n_jobs=-1, metric='dtw', metric_params=dtw_params,
                                                                   n_clusters=n_clusters)},
                  2: {'name': 'K-shape', 'metric': 'cross-correlation',
                      'model': lambda n_clusters: KShape(max_iter=max_iter, random_state=RAND_SEED,
                                                         n_clusters=n_clusters)}}

    # Instantiation of timeseriesclustering object
    tsc = TimeSeriesClustering(X=X_data, max_iter=max_iter, dtw_params=dtw_params)

    # Validation to choose optimal algorithm and number of clusters
    # Range of the number of clusters to try from
    perc_col_removed = 0.1
    val_idx_dict = {0: {'name': 'silhouette', 'func': silhouette_index, 'lower_better': False},
                    1: {'name': 'dunn', 'func': dunn_index, 'lower_better': True},
                    2: {'name': 'davies-bouldin', 'func': davies_bouldin_index, 'lower_better': True},
                    3: {'name': 'calinski-harabasz', 'func': calinski_harabasz_index, 'lower_better': True},
                    4: {'name': 'apn', 'func': stability_index, 'stability_params': {'method': 'apn', 'perc_col_removed': perc_col_removed}, 'lower_better': True},
                    5: {'name': 'ad', 'func': stability_index, 'stability_params': {'method': 'ad', 'perc_col_removed': perc_col_removed}, 'lower_better': True},
                    6: {'name': 'hartigan', 'func': hartigan_index, 'hartigan_params': {}, 'lower_better': True}}
    k1, k2 = 2, 6
    val_tsc = ValidationTimeSeriesClustering(X=X_data, k1=k1, k2=k2, max_iter=max_iter,
                                             algos_dict=algos_dict, val_idx_dict=val_idx_dict)
    optim_algo, optim_n_clusters = val_tsc.get_best_algo()
    # Save output of validation
    val_tsc.save_output_to_file('output_validation.txt', optim_algo, optim_n_clusters)

    # Run clustering with optimal algorithm and number of clusters
    tsc.run_timeseries_clustering(optim_algo, optim_n_clusters)

    # Plot time series clusters with centroids and % insights
    tsc.plot_timeseries_clustering(algo=optim_algo, n_clusters=optim_n_clusters, df_insight=df_insight)
