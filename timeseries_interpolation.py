import numpy as np
import pandas as pd
import logging

# HYPER PARAMS
TARGET_N = 2050
ONLY_INSIGHT = False


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info('Loading file...')
data = pd.read_csv("dataRaw_AnswerUpdated.csv")
# REMOVE INCORRECT TRIALS
idx_to_drop = data[data['correct'] == 0].index
data.drop(idx_to_drop, inplace=True)
# REMOVE TRIALS SOLVED THROUGH NEITHER INSIGHT NOR NON-INSIGHT
logging.info('Removing trials solved with strategy 3...')
idx_to_drop = data[data['solution_strategy_response.keys'] == 3].index
data.drop(idx_to_drop, inplace=True)
# REMOVE NON-INSIGHT TRIALS
if ONLY_INSIGHT:
    logging.info('Removing trials solved with strategy 2...')
    idx_to_drop = data[data['solution_strategy_response.keys'] == 2].index
    data.drop(idx_to_drop, inplace=True)

# CAP > 10 VALUES AT 10
data.loc[data['rating'] > 10,'rating'] = 10.0

# ADD UNIQUE TRIAL NUMBERS
logging.info('Adding unique trials refs...')
data['unique_trial'] = data['Id'].astype(str) + data['Trial'].astype(str)
data['unique_trial'] = pd.factorize(data['unique_trial'])[0] + 1
# REMOVE DUPLICATES (i.e. if two data points within a trial have the same rating at the same time)
data = data.drop_duplicates(subset=['rating', 'time', 'unique_trial'])

# BUILD UPSAMPLED DATAFRAME
logging.info('Up sampling time series...')
tot_trial_nb = data['unique_trial'].nunique()
new_df = pd.DataFrame(columns=['Id', 'InterpRating', 'solution_strategy_response.keys'])
for trial in range(1, tot_trial_nb + 1):
    logging.info(f'TRIAL N°{trial}...')
    # get time series of trial
    y_data = data[data['unique_trial'] == trial]['rating']
    # get new indices of values
    target_indices = np.arange(0, TARGET_N, TARGET_N / len(y_data))
    round_target_indices = [np.floor(idx) for idx in target_indices]
    if len(round_target_indices) != len(y_data):
        delta = len(round_target_indices) - len(y_data)
        # if number of indices higher than number of data points (due to rounding), remove the excess indices randomly
        if delta > 0:
            for i in range(delta):
                round_target_indices.pop(np.random.randint(0,  len(round_target_indices)-1))
        # since rounding is down, the number of indices shouldn't be lower
        elif delta < 0:
            ValueError("Number of target indices smaller than data points!")
    if len(round_target_indices) != len(y_data):
        ValueError("Correction of target indices did not work!")
    # assign values to target indices
    logging.info('Stretching data...')
    y_stretched = y_data.set_axis(round_target_indices).reindex(range(0, TARGET_N))
    # interpolate the missing values
    logging.info('Interpolating data...')
    y_interp = y_stretched.interpolate(method='index', limit_direction='both')
    # add time series to new df
    logging.info('Add time series to df...')
    id = data[data['unique_trial'] == trial]['Id'].unique().item()
    sol_strat = data[data['unique_trial'] == trial]['solution_strategy_response.keys'].unique().item()
    interp_data = pd.DataFrame(
        {'Id': [id] * len(y_interp), 'Trial': [trial] * len(y_interp), 'InterpRating': y_interp,
         'solution_strategy_response.keys': [sol_strat] * len(y_interp)})
    new_df = pd.concat([new_df, interp_data])

# save new dataframe to csv file
if ONLY_INSIGHT:
    new_df.to_csv("dataRaw_AnswerUpdated_processed_INSIGHT.csv")
else:
    new_df.to_csv("dataRaw_AnswerUpdated_processed.csv")

