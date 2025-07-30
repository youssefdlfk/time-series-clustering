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
    X_data, df_insight = data_proc_insight(csv_name=config.csv_name, timeseries_length=config.ts_length,
                                           down_sample_factor=config.down_sample_factor, filter=None)

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

    # Step 6: Run clustering with best model
    logging.info("Running clustering for best model...")
    # Run clustering with optimal algorithm and number of clusters
    model, labels = clusterer.run_timeseries_clustering(optim_algo, optim_n_clusters)
    # Step 7: Save best model
    logging.info("Saving validation results...")
    with open('optim_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    np.save('optim_labels.npy', labels)

    # Step 8: Plot time series clusters with centroids and % insights of best model
    clusterer.plot_timeseries_clustering(model=model, labels=labels, algo_str=optim_algo, n_clusters=optim_n_clusters,
                                         df_insight=df_insight)
    # Step 9: Save all outputs
    logging.info("Saving all validation results in csv files...")
    save_outputs_to_csv(df_insight, config, clusterer, validator)

    logging.info("Done!")


if __name__ == '__main__':
    main()
