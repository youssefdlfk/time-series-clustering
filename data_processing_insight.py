import numpy as np
import pandas as pd
import torch
import copy
import pickle
import logging

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
        print(e)

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



def save_outputs_to_csv(df_insight):
    # Load data
    optim_labels = np.load('results_hpc_downsampling2/optim_labels.npy')
    score_matrix = np.load('results_hpc_downsampling2/score_matrix.npy')
    rank_matrix = np.load('results_hpc_downsampling2/rank_matrix.npy')
    with open('results_hpc_downsampling2/optim_model.pkl', 'rb') as f:
        optim_model = pickle.load(f)

    # Some columns and rows
    indices_cols = ['silhouette', 'dunn', 'davies-bouldin', 'calinski-harabasz', 'apn', 'ad', 'hartigan']
    algo_clus_rows = ['euclidean_2clusters', 'euclidean_3clusters', 'euclidean_4clusters', 'euclidean_5clusters',
             'euclidean_6clusters', 'dtw_2clusters', 'dtw_3clusters', 'dtw_4clusters', 'dtw_5clusters', 'dtw_6clusters',
             'kshape_2clusters', 'kshape_3clusters', 'kshape_4clusters', 'kshape_5clusters', 'kshape_6clusters']
    algo_clus_col = 'algorithm_cluster'

    # Dataframes
    score_df = pd.DataFrame(score_matrix, columns=indices_cols)
    score_df.insert(loc=0, column=algo_clus_col, value=algo_clus_rows)
    labels_df = copy.deepcopy(df_insight)
    labels_df['Cluster'] = optim_labels
    rank_df = pd.DataFrame(rank_matrix, columns=indices_cols)
    rank_df.insert(loc=0, column=algo_clus_col, value=algo_clus_rows)
    cluster_center_df = pd.DataFrame(optim_model.cluster_centers_.squeeze(-1).transpose(), columns=['cluster0', 'cluster1'])

    # Saving to .csv files
    labels_df.to_csv('cluster_labels.csv')
    score_df.to_csv('score_matrix.csv')
    rank_df.to_csv('rank_matrix.csv')
    cluster_center_df.to_csv('cluster_centers.csv')






