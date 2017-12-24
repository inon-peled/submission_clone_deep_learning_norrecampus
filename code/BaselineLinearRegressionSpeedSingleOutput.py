from speed_data_maker import default_bootstrap_speed_data_maker
from common import EXAMPLE_PLACE_ID, compute_error_statistics
from sklearn.linear_model import LinearRegression


class BaselineLinearRegressionSpeedSingleOutput(object):
    def __init__(self, speed_data_maker):
        self.speed_data_maker = speed_data_maker

    @classmethod
    def name(cls):
        return 'LR'

    def baseline_errors_statistics(self):
        X_train_normalized, Y_train_normalized, X_test_normalized, _ = \
            self.speed_data_maker.get_train_and_test_inputs()
        trained_lr = LinearRegression(fit_intercept=False)\
            .fit(X_train_normalized.values, Y_train_normalized.values)
        lr_predictions_normalized = trained_lr.predict(X_test_normalized)
        errors_df = self.speed_data_maker.individual_errors_without_interpolated_values(lr_predictions_normalized)
        return compute_error_statistics(errors_df, 'speed_km_hr_true', 'speed_km_hr_predicted')


if __name__ == '__main__':
    print(BaselineLinearRegressionSpeedSingleOutput(default_bootstrap_speed_data_maker(EXAMPLE_PLACE_ID))\
        .baseline_errors_statistics())