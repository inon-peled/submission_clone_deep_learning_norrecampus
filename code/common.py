import pandas as pd
import os
import numpy as np


BEST_NUM_PAST_LAGS = 3
BEST_NUM_EPOCHS = 100
BEST_MINI_BATCH_SIZE = 256
BEST_LSTM_STATE_SIZE = 30

SPLIT_DATE = '2015-06-01'
NUM_LAGS = 13
THRESHOLD_SPEED_OUTLIER_KM_PER_HR = 110
EXAMPLE_PLACE_ID = 'ChIJZaR1M1hSUkYRxP0WkwYYy_k'
DATA_PATH = os.path.join('.', 'data', 'by_place_5min')


def compute_error_statistics(errors_df, column_name_true_values, column_name_predicted_values):
    abs_errors = errors_df.error.abs()
    return {
        'corr': np.corrcoef(errors_df[column_name_predicted_values], errors_df[column_name_true_values])[0, 1],
        'mae': np.mean(abs_errors),
        'rmse': np.sqrt(np.mean(abs_errors ** 2)),
        'r2': max(0, 1 - np.sum(abs_errors ** 2) /
                  np.sum((errors_df[column_name_true_values] - np.mean(errors_df[column_name_true_values])) ** 2))
    }


def get_lat_lng(place_ids):
    return pd.read_csv('place_details.csv', index_col='place_id')[lambda df: df.index.isin(place_ids)]
