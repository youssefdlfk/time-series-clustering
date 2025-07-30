"""
Data preprocessing module for time series clustering.

Specifically tailored to process closeness-to-solution trajectories from psychological insight experiments.
Handles loading, reshaping, downsampling, and labeling data for subsequent clustering analysis.
"""


import copy
import logging
import pickle

import numpy as np
import pandas as pd
import torch

from ts_clustering.clustering.utils import spearman_footrule_distance


def data_proc_insight(csv_name: str, timeseries_length: int, down_sample_factor: int, filter: str) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Process CSV data from Insight experiment and convert it into a tensor for clustering
    :param down_sample_factor: Down sampling factor for time series data
    :param csv_name: File name of the CSV data
    :param timeseries_length: Number of data points of each time series (should be the same)
    :return: A tensor of the time series data and a dataframe indicating insight or not insight for each time series
    """
    # Load data
    try:
        df = pd.read_csv(csv_name)
    except FileNotFoundError as e:
        logging.error(e)
        raise SystemExit(f"File {csv_name} not found. Exiting.")

    # Check if required columns are present
    required_columns = ['Id', 'Trial', 'InterpRating', 'solution_strategy_response.keys']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f'Error: Missing required column {col} in the CSV file.')

    # Rename for clarity and easier access
    df = df.rename(columns={'solution_strategy_response.keys': 'Insight'})

    # Filter insight or non-insight trials if specified
    if filter is not None:
        if filter == 'insight':
            logging.info("Keeping only insight trials...")
            df = df[df['Insight'] == 1]
        elif filter == 'non-insight':
            logging.info("Keeping only non-insight trials...")
            df = df[df['Insight'] == 2]
        else:
            raise ValueError(f'{filter} is not recognized as filter.')

    # Count number of trials per participant
    trial_counts = df.groupby(['Id'])['Trial'].nunique().reset_index(name='Trial_count')

    # Initialize list to hold participants' trials
    X_list = []

    # Construct list with participant data (1 tensor in the list = all trials of 1 participant)
    for specific_id in trial_counts['Id']:
        nb_trials_id = trial_counts[trial_counts['Id'] == specific_id]['Trial_count'].values[0]
        X_data = torch.from_numpy(df[df['Id'] == specific_id]['InterpRating'].values.astype('float32'))
        X_data = torch.reshape(X_data, (nb_trials_id, timeseries_length))
        X_list.append(X_data)

    # Concatenate tensors into a single one containing all trials
    dim1 = trial_counts['Trial_count'].sum()
    X_data = torch.empty(dim1, timeseries_length)
    first_idx = 0
    for i in range(len(X_list)):
        second_idx = first_idx+X_list[i].shape[0]
        X_data[first_idx:second_idx] = X_list[i]
        first_idx = second_idx

    # Create dataframe indicating whether insight or non insight for each trial
    df_insight = df.drop_duplicates(subset=['Id', 'Trial', 'Insight'])[['Id', 'Trial', 'Insight']]

    # Convert tensor to numpy array
    X_data = X_data.numpy()

    # Downsampling
    X_data = X_data[:, ::down_sample_factor]

    return X_data, df_insight


def save_outputs_to_csv(df_insight, config, clusterer, validator):
    # Load data
    optim_labels = np.load('optim_labels.npy')
    score_matrix = np.load('score_matrix.npy')
    rank_matrix = np.load('rank_matrix.npy')
    with open('optim_model.pkl', 'rb') as f:
        optim_model = pickle.load(f)

    # Some columns and rows
    indices_cols = list(validator.val_idx_dict.keys())
    algo_clus_rows = [algo + '_' + str(n_cluster) + 'clusters' for algo in clusterer.algos_dict.keys() for n_cluster in range(config.k1, config.k2+1)]
    clus_cent_col =['cluster' + str(nb) for nb in range(optim_model.n_clusters)]
    algo_clus_col = 'algorithm_cluster'

    # Dataframes
    score_df = pd.DataFrame(score_matrix, columns=indices_cols)
    score_df.insert(loc=0, column=algo_clus_col, value=algo_clus_rows)
    labels_df = copy.deepcopy(df_insight)
    labels_df['Cluster'] = optim_labels
    rank_df = pd.DataFrame(rank_matrix, columns=indices_cols)
    rank_df.insert(loc=0, column=algo_clus_col, value=algo_clus_rows)
    cluster_center_df = pd.DataFrame(optim_model.cluster_centers_.squeeze(-1).transpose(), columns=clus_cent_col)

    # Final ranking
    nb_val_idx = len(indices_cols)
    ranking_ref = np.zeros(nb_val_idx)
    dist_to_ref = []
    for algo_clus_idx in range(score_matrix.shape[0]):
        dist_to_ref.append(spearman_footrule_distance(ranking_ref, rank_matrix[algo_clus_idx, :]))
    dist_to_ref_df = pd.DataFrame(dist_to_ref, columns=['Distance to ideal vector'])
    dist_to_ref_df.insert(loc=0, column=algo_clus_col, value=algo_clus_rows)

    # Saving to .csv files
    labels_df.to_csv(config.labels_output_file, index=False)
    score_df.to_csv(config.score_matrix_output_file, index=False)
    rank_df.to_csv(config.rank_matrix_output_file, index=False)
    dist_to_ref_df.to_csv(config.dist_to_ref_output_file, index=False)
    cluster_center_df.to_csv(config.cluster_centers_output_file, index=False)






