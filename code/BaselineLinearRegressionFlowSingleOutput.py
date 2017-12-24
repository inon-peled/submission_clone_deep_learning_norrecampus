from flow_data_maker import default_bootstrap_flow_data_maker
from common import EXAMPLE_PLACE_ID, compute_error_statistics
from sklearn.linear_model import LinearRegression


class BaselineLinearRegressionFlowSingleOutput(object):
    def __init__(self, speed_data_maker):
        self.flow_data_maker = speed_data_maker

    @classmethod
    def name(cls):
        return 'LROnlyFlow'

    def baseline_errors_statistics(self):
        x_train, y_train, x_test, _ = self.flow_data_maker.get_train_and_test_inputs()
        trained_lr = LinearRegression(fit_intercept=False).fit(x_train.values, y_train.values)
        lr_predictions = trained_lr.predict(x_test).round()
        errors_df = self.flow_data_maker.individual_errors_without_interpolated_values(lr_predictions)
        return compute_error_statistics(errors_df, 'flow_decile_true', 'flow_decile_predicted')


if __name__ == '__main__':
    print(BaselineLinearRegressionFlowSingleOutput(default_bootstrap_flow_data_maker(EXAMPLE_PLACE_ID))\
        .baseline_errors_statistics())
