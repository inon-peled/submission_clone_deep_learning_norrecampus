from common import *
import pandas as pd

from flow_data_maker import FlowDataMaker
from speed_data_maker import SpeedDataMaker


class SeparateSpeedAndFlowDataMaker(object):
    def __init__(self, default_speed_if_still_missing_after_interpolation,
                 default_flow_if_still_missing_after_interpolation,
                 threshold_high_speed_outlier,
                 place_id, split_date, num_lags):
        self.place_id = place_id
        self.speed_data_maker = SpeedDataMaker(default_speed_if_still_missing_after_interpolation,
                                               place_id, split_date, threshold_high_speed_outlier, num_lags)
        self.flow_data_maker = FlowDataMaker(default_flow_if_still_missing_after_interpolation,
                                             place_id, split_date, num_lags)

    def get_train_and_test_inputs(self):
        speed_inputs = self.speed_data_maker.get_train_and_test_inputs()
        flow_inputs = self.flow_data_maker.get_train_and_test_inputs()
        return zip(speed_inputs, flow_inputs)

    def individual_errors_without_interpolated_values(self, speed_predictions_normalized, flow_predictions):
        return self.speed_data_maker.individual_errors_without_interpolated_values(speed_predictions_normalized), \
               self.flow_data_maker.individual_errors_without_interpolated_values(flow_predictions)

    def stats(self, predictions_speed, predictions_flow):
        errors_df_speed = self.speed_data_maker.individual_errors_without_interpolated_values(predictions_speed)
        errors_df_flow = self.flow_data_maker.individual_errors_without_interpolated_values(predictions_flow)
        speed_errors_agg = compute_error_statistics(errors_df_speed, 'speed_km_hr_true', 'speed_km_hr_predicted')
        flow_errors_agg = compute_error_statistics(errors_df_flow, 'flow_decile_true', 'flow_decile_predicted')
        return pd.DataFrame({k + '_speed': v for k, v in speed_errors_agg.items()}, index=[self.place_id]).join(
            pd.DataFrame({k + '_flow': v for k, v in flow_errors_agg.items()}, index=[self.place_id]))


def default_bootstrap_separate_speed_and_flow_data_maker(place_id, num_lags):
    return SeparateSpeedAndFlowDataMaker(0, 0, THRESHOLD_SPEED_OUTLIER_KM_PER_HR, place_id, SPLIT_DATE, num_lags)


if __name__ == '__main__':
    data_maker = default_separate_bootstrap_speed_and_flow_data_maker(EXAMPLE_PLACE_ID, 1)
    x_train, y_train, x_test, y_test = data_maker.get_train_and_test_inputs()
    print(x_train[0].head(), '\n')
    print(x_train[1].head(), '\n')
    print(y_test[1].head(), '\n')
