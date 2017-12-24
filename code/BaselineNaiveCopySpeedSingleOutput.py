import numpy as np
from speed_data_maker import default_bootstrap_speed_data_maker
from common import EXAMPLE_PLACE_ID, compute_error_statistics


class BaselineNaiveCopySpeedSingleOutput(object):
    def __init__(self, speed_data_maker):
        self.speed_data_maker = speed_data_maker

    @classmethod
    def name(cls):
        return 'NaiveCopy'

    def baseline_errors_statistics(self):
        X_train_normalized, Y_train_normalized, X_test_normalized, Y_test_normalized = \
            self.speed_data_maker.get_train_and_test_inputs()
        predictions_normalized = np.insert(Y_test_normalized.values, 0, Y_train_normalized[-1])[:-1]
        errors_df = self.speed_data_maker.individual_errors_without_interpolated_values(predictions_normalized)
        return compute_error_statistics(errors_df, 'speed_km_hr_true', 'speed_km_hr_predicted')


if __name__ == '__main__':
    print(BaselineNaiveCopySpeedSingleOutput(default_bootstrap_speed_data_maker(EXAMPLE_PLACE_ID))\
        .baseline_errors_statistics())
