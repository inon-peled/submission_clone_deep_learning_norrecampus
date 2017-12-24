from speed_and_flow_data_maker import default_bootstrap_speed_and_flow_data_maker
from common import EXAMPLE_PLACE_ID, compute_error_statistics
from sklearn.linear_model import LinearRegression


class BaselineLinearRegressionSpeedsAndFlows(object):
    def __init__(self, speed_and_flow_data_maker):
        self.speed_and_flow_data_maker = speed_and_flow_data_maker

    @classmethod
    def name(cls):
        return 'LRSpeedsAndFlows'

    def baseline_errors_statistics(self):
        x_train, y_train, x_test, _ = self.speed_and_flow_data_maker.get_train_and_test_inputs()
        trained_lr = LinearRegression(fit_intercept=False).fit(x_train.values, y_train.values)
        lr_predictions = trained_lr.predict(x_test)
        speed_errors_df, flow_errors_df = self.speed_and_flow_data_maker\
            .individual_errors_without_interpolated_values(lr_predictions[:, 0], lr_predictions[:, 1])
        return compute_error_statistics(speed_errors_df, 'speed_km_hr_true', 'speed_km_hr_predicted'), \
               compute_error_statistics(flow_errors_df, 'flow_decile_true', 'flow_decile_predicted')


if __name__ == '__main__':
    print(BaselineLinearRegressionSpeedsAndFlows(default_bootstrap_speed_and_flow_data_maker(EXAMPLE_PLACE_ID))\
        .baseline_errors_statistics())
