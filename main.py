import logging
import pickle

import numpy as np

from config import ClusteringConfig
from data_processing_insight import data_proc_insight
from timeseries_clustering import TimeSeriesClustering
from validation_clustering import ValidationTimeSeriesClustering

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    config = ClusteringConfig(ts_length=2100,
                              csv_name="2100interpolationfulldata.csv",
                              max_iter=1,
                              window_size_perc=0.05,
                              perc_col_del=0.1,
                              k1=2,
                              k2=3,
                              down_sample_factor=7)

    # Step 1: Load and process time series data and extract dataframe on insight or not
    logging.info("Loading and preprocessing data...")
    X_data, df_insight = data_proc_insight(csv_name=config.csv_name, timeseries_length=config.ts_length,
                                           down_sample_factor=config.down_sample_factor)

    # Step 2: Initialize Clustering and Validation
    clusterer = TimeSeriesClustering(X=X_data, config=config)
    validator = ValidationTimeSeriesClustering(X=X_data, config=config)

    # Step 3: Find best combination of algorithm and number of clusters
    logging.info("Performing validation...")
    optim_algo, optim_n_clusters = validator.get_best_algo()

    # Step 4: Save output of Validation
    logging.info("Saving validation results...")
    # Save output of validation
    validator.save_output_to_file('output_validation.txt', optim_algo, optim_n_clusters)

    # Step 5: Run clustering with best model
    logging.info("Running clustering for best model...")
    # Run clustering with optimal algorithm and number of clusters
    model, labels = clusterer.run_timeseries_clustering(optim_algo, optim_n_clusters)
    # Step 6: Save best model
    logging.info("Saving validation results...")
    with open('optim_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    np.save('optim_labels.npy', labels)

    # Step 7: Plot time series clusters with centroids and % insights of best model
    clusterer.plot_timeseries_clustering(model=model, labels=labels, algo_str=optim_algo, n_clusters=optim_n_clusters,
                                         df_insight=df_insight)

if __name__ == '__main__':
    main()
