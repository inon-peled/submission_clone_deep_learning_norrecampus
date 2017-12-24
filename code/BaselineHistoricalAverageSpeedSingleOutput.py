import numpy as np
from speed_data_maker import default_bootstrap_speed_data_maker
from common import EXAMPLE_PLACE_ID, compute_error_statistics
from sklearn.linear_model import LinearRegression


class BaselineHistoricalAverageSpeedSingleOutput(object):
    def __init__(self, speed_data_maker):
        self.speed_data_maker = speed_data_maker

    @classmethod
    def name(cls):
        return 'HistoricalAverage'

    def baseline_errors_statistics(self):
        X_train_normalized, Y_train_normalized, X_test_normalized, Y_test_normalized = \
            self.speed_data_maker.get_train_and_test_inputs()
        cum_avg_all = np.cumsum(np.concatenate((Y_train_normalized.values, Y_test_normalized.values))) / \
                      np.arange(1, len(Y_train_normalized) + len(Y_test_normalized) + 1)
        predictions_normalized = cum_avg_all[len(Y_train_normalized) - 1: -1]
        errors_df = self.speed_data_maker.individual_errors_without_interpolated_values(predictions_normalized)
        return compute_error_statistics(errors_df, 'speed_km_hr_true', 'speed_km_hr_predicted')


if __name__ == '__main__':
    print(BaselineHistoricalAverageSpeedSingleOutput(default_bootstrap_speed_data_maker(EXAMPLE_PLACE_ID))\
        .baseline_errors_statistics())