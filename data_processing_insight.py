import numpy as np
import pandas as pd
import torch


def data_proc_insight(csv_name: str, timeseries_length: int, down_sample_factor: int) -> tuple[np.ndarray, pd.DataFrame]:
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
