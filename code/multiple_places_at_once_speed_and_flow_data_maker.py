from place_groups import selected_middle_of_roads, selected_junctions
from common import *
import pandas as pd
import numpy as np

from speed_and_flow_data_maker import SpeedAndFlowDataMaker


class MultiplePlacesAtOnceSpeedAndFlowDataMaker(object):
    def __init__(self,
                 default_speed_if_still_missing_after_interpolation,
                 default_flow_if_still_missing_after_interpolation,
                 threshold_high_speed_outlier,
                 place_group, split_date, num_lags):
        self.place_group_sorted = sorted(place_group)
        self.speed_and_flow_data_makers = [
            SpeedAndFlowDataMaker(default_speed_if_still_missing_after_interpolation,
                                  default_flow_if_still_missing_after_interpolation,
                                  threshold_high_speed_outlier,
                                  place_id,
                                  split_date,
                                  num_lags)
            for place_id in self.place_group_sorted]

    def get_train_and_test_inputs(self):
        inputs = [data_maker.get_train_and_test_inputs()
                  for data_maker in self.speed_and_flow_data_makers]
        return tuple(np.swapaxes(np.concatenate([np.expand_dims(inp[i].sort_index(), 0) for inp in inputs]), 0, 1)
                     for i in range(4))

    def individual_errors_without_interpolated_values(self, speed_predictions_normalized, flow_predictions):
        return self.speed_data_maker.individual_errors_without_interpolated_values(speed_predictions_normalized), \
               self.flow_data_maker.individual_errors_without_interpolated_values(flow_predictions)

    def stats(self, predictions_speed, predictions_flow):
        return pd.concat([self.speed_and_flow_data_makers[i].stats(predictions_speed[:, i], predictions_flow[:, i])
                for i in range(len(self.place_group_sorted))])


def default_bootstrap_multiple_places_at_once_speed_and_flow_data_maker(places_group, num_lags):
    return MultiplePlacesAtOnceSpeedAndFlowDataMaker(
        0, 0, THRESHOLD_SPEED_OUTLIER_KM_PER_HR, places_group, SPLIT_DATE, num_lags)


if __name__ == '__main__':
    data_maker = default_bootstrap_multiple_places_at_once_speed_and_flow_data_maker(
        (selected_junctions + selected_middle_of_roads)[::4], BEST_NUM_PAST_LAGS)
    x_train, y_train, x_test, y_test = data_maker.get_train_and_test_inputs()
    print(('------------------ X_train %s: ---------------\n' % (x_train.shape,)), x_train)
    print(('------------------ Y_test: %s ---------------\n' % (y_test.shape,)), y_test)
