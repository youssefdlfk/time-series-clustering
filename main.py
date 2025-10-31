"""
Main entry point for the Time Series Clustering project.

This script orchestrates the workflow including data preprocessing, algorithm validation, clustering execution,
and saving of results.
"""

import logging
import pickle

import numpy as np

from config import default_config as config
from ts_clustering.clustering.timeseries_clustering import TimeSeriesClustering
from ts_clustering.data_processing_insight import (data_proc_insight,
                                                   save_outputs_to_csv)
from ts_clustering.validation.validation_clustering import \
    ValidationTimeSeriesClustering

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():

    # Step 1: Load and process time series data and extract dataframe on insight or not
    logging.info("Loading and preprocessing data...")
    # Format data for clustering and convert to tensor
    X_data, df_insight = data_proc_insight(csv_name=config.csv_name, timeseries_length=config.ts_length,
                                           down_sample_factor=config.down_sample_factor, filter=None)

    # Step 2: Initialize Clustering and Validation
    clusterer = TimeSeriesClustering(X=X_data, config=config)
    validator = ValidationTimeSeriesClustering(X=X_data, config=config)

    # Step 3: Find best combination of algorithm and number of clusters
    logging.info("Performing validation...")
    topk_algo_clus_list = validator.get_topk_algo(topk=config.topk)

    # Step 4: Save output of Validation
    logging.info(f"Saving validation results for top {config.topk} models...")

    # Save output of validation for top k models
    for i, (optim_algo, optim_n_clusters) in enumerate(topk_algo_clus_list):
        validator.save_output_to_file('saved_outputs/output_validation.txt', optim_algo.name, optim_n_clusters)
         # Step 6: Run clustering with top k models
        logging.info(f"Running clustering for top {i+1} model...")
        # Run clustering with optimal algorithm and number of clusters
        optim_model, optim_labels = clusterer.run_timeseries_clustering(optim_algo, optim_n_clusters)

        # Step 7: Save best model
        logging.info(f"Saving validation results for model {i+1}...")
        with open(f'saved_outputs/optim_model_{i+1}.pkl', 'wb') as file:
            pickle.dump(optim_model, file)
        np.save(f'saved_outputs/optim_labels_{i+1}.npy', optim_labels)

        # Step 8: Plot time series clusters with centroids and % insights of best model
        clusterer.plot_timeseries_clustering(model=optim_model, labels=optim_labels, algo=optim_algo, n_clusters=optim_n_clusters,
                                         df_insight=df_insight, k_model=i+1)
        # Step 9: Save all outputs
        logging.info(f"Saving all validation results in csv files for model {i+1}...")
        save_outputs_to_csv(i+1, df_insight, config, clusterer, validator)

    logging.info("Done!")


if __name__ == '__main__':
    main()
