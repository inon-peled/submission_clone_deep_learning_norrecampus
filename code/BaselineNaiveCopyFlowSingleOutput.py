import numpy as np
from flow_data_maker import default_bootstrap_flow_data_maker
from common import EXAMPLE_PLACE_ID, compute_error_statistics


class BaselineNaiveCopyFlowSingleOutput(object):
    def __init__(self, flow_data_maker):
        self.flow_data_maker = flow_data_maker

    @classmethod
    def name(cls):
        return 'NaiveCopy'

    def baseline_errors_statistics(self):
        x_train, y_train, x_test, y_test = self.flow_data_maker.get_train_and_test_inputs()
        predictions = np.insert(y_test.values, 0, y_train[-1])[:-1].round()
        errors_df = self.flow_data_maker.individual_errors_without_interpolated_values(predictions)
        return compute_error_statistics(errors_df, 'flow_decile_true', 'flow_decile_predicted')


if __name__ == '__main__':
    print(BaselineNaiveCopyFlowSingleOutput(default_bootstrap_flow_data_maker(EXAMPLE_PLACE_ID))\
        .baseline_errors_statistics())
