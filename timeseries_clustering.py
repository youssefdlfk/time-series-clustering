import numpy as np
import matplotlib.pyplot as plt
from data_processing_insight import data_proc_insight
from tslearn.clustering import TimeSeriesKMeans, KShape
from validation_clustering import ValidationTimeSeriesClustering


class TimeSeriesClustering:
    def __init__(self, X, max_iter, params_dtw, algos_dict, val_idx_dict):
        self.X = X
        self.max_iter = max_iter
        self.params_dtw = params_dtw
        self.algos_dict = algos_dict
        self.val_idx_dict = val_idx_dict

    def validate_clustering(self, k1, k2):
        """
        Find the optimal algorithm and number of clusters according to a set of validation indices and a voting rule.
        :param k1: Minimum number of clusters to test.
        :param k2: Maximum number of clusters to test.
        :return: Optimal algorithm name and optimal number of clusters
        """
        val_tsc = ValidationTimeSeriesClustering(X=self.X, k1=k1, k2=k2, algos_dict=self.algos_dict, val_idx_dict=self.val_idx_dict)
        optim_algo, optim_n_clusters = val_tsc.get_best_algo()
        return optim_algo, optim_n_clusters

    @staticmethod
    def save_validation_output(optim_algo, optim_n_clusters):
        """
        Save the output of the clustering validation in .txt file.
        :param optim_algo: Optimal algorithm name
        :param optim_n_clusters: Optimal number of clusters within the tested range
        """
        with open("validation_output.txt", "w") as text_file:
            text_file.write(f'Algorithm: {optim_algo}, N° clusters: {optim_n_clusters}')

    def run_timeseries_clustering(self, algo, n_clusters):
        """
        Run a clustering algorithm.
        :param algo: Algorithm
        :param n_clusters: Number of clusters
        :return:
        """
        if algo == 'euclidean':
            model = TimeSeriesKMeans(n_clusters=n_clusters, init='k-means++', max_iter=self.max_iter, metric='euclidean', random_state=42)
        elif algo == 'dtw':
            model = TimeSeriesKMeans(n_clusters=n_clusters, init='k-means++', max_iter=self.max_iter, random_state=42,
                                 n_jobs=-1, metric='dtw', metric_params=self.params_dtw)
        elif algo == 'kshape':
            model = KShape(n_clusters=n_clusters, max_iter=self.max_iter, random_state=42)
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
    params_dtw = {'global_constraint': "sakoe_chiba", 'sakoe_chiba_radius': window_size}
    down_sample_factor = 6
    max_iter = 10

    # Down sampling of the time series data
    X_data = X_data[:, ::down_sample_factor]

    # Dictionaries of the algorithms and indices used for validation
    algos_dict = {0: 'euclidean', 1: 'dtw', 2: 'kshape'}
    val_idx_dict = {0: 'silhouette', 1: 'dunn', 2: 'davies-bouldin', 3: 'calinski-harabasz', 4: 'apn', 5: 'ad',
                    6: 'hartigan'}

    # Instantiation of timeseriesclustering object
    tsc = TimeSeriesClustering(X=X_data, max_iter=max_iter, params_dtw=params_dtw, algos_dict=algos_dict,
                               val_idx_dict=val_idx_dict)

    # Validation to choose optimal algorithm and number of clusters
    # Range of the number of clusters to try from
    k1, k2 = 2, 6
    optim_algo, optim_n_clusters = tsc.validate_clustering(k1=k1, k2=k2)

    # Run clustering with optimal algorithm and number of clusters
    tsc.run_timeseries_clustering(optim_algo, optim_n_clusters)

    # Plot time series clusters with centroids and % insights
    tsc.plot_timeseries_clustering(algo=optim_algo, n_clusters=optim_n_clusters, df_insight=df_insight)
