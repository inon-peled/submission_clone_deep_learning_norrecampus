from common import *

import os
from datetime import timedelta
import pandas as pd


def _split_timestamp(ser):
    return pd.DataFrame(ser).assign(day_of_week=lambda df: df.index.dayofweek,
                                    time_of_day=lambda df: df.index.time).reset_index()


class FlowDataMaker(object):
    def __init__(self, default_flow_if_still_missing_after_interpolation, place_id, split_date, num_lags_back):
        self.default_flow_if_still_missing_after_interpolation = default_flow_if_still_missing_after_interpolation
        self.place_id = place_id
        self.split_date = split_date
        self.num_lags_back = num_lags_back

    def _imputation_for_missing_values(self, ser):
        missing_timestamps = pd.date_range(min(ser.index.date), max(ser.index.date) + timedelta(days=1),
                                           freq='5T').difference(ser.index)
        df_with_nans_where_missing = pd.DataFrame(ser).join(pd.DataFrame(index=missing_timestamps), how='outer').assign(
            original_value=lambda df: df.iloc[:, 0])
        flows_interpolated = df_with_nans_where_missing.iloc[:, 0] \
            .interpolate() \
            .fillna(self.default_flow_if_still_missing_after_interpolation) \
            .round() \
            .astype(int) \
            .to_frame() \
            .assign(original_value=df_with_nans_where_missing.original_value) \
            .assign(is_interpolated=lambda df: df.iloc[:, 0] != df.iloc[:, 1])
        assert (max(flows_interpolated.flow_decile) <= 9) and (min(flows_interpolated.flow_decile) >= 0)
        return flows_interpolated

    def _get_df(self):
        return pd.read_csv(
            os.path.join(DATA_PATH, self.place_id + '.csv'),
            parse_dates=['start_interval_s', 'end_interval_s']
        )[lambda df: df.start_interval_s >= '2015-01-01'] \
            .rename(columns={'start_interval_s': 't', 'flow_bucket': 'flow_decile'}) \
            .set_index('t') \
            .flow_decile

    def get_train_and_test_inputs(self):
        flows_interpolated = self._imputation_for_missing_values(self._get_df()).flow_decile
        lags = pd.concat([flows_interpolated.shift(x).rename('f%d'% x)
                          for x in range(self.num_lags_back + 1)], axis=1)[self.num_lags_back:]
        train = lags[lags.index < self.split_date]
        x_train = train.iloc[:, 1:]
        y_train = train.iloc[:, 0]
        test = lags[lags.index >= self.split_date]
        x_test = test.iloc[:, 1:]
        y_test = test.iloc[:, 0]
        return x_train, y_train, x_test, y_test

    def individual_errors_without_interpolated_values(self, predictions):
        _, _, _, ser_y_test = self.get_train_and_test_inputs()
        ser_predictions = pd.Series(predictions, index=ser_y_test.index, name='flow_decile')
        interpolated_timestamps = self._imputation_for_missing_values(self._get_df())[
            lambda df: df.is_interpolated].index
        return ser_y_test \
            .rename('flow_decile')\
            .to_frame()\
            .join(ser_predictions.to_frame(), lsuffix='_true', rsuffix='_predicted')\
            .loc[lambda df: df.index.difference(interpolated_timestamps)]\
            .assign(error=lambda df: df.flow_decile_true - df.flow_decile_predicted)

    def stats(self, predictions):
        errors_df = self.individual_errors_without_interpolated_values(predictions.round())
        return compute_error_statistics(errors_df, 'flow_decile_true', 'flow_decile_predicted')


def default_bootstrap_flow_data_maker(place_id, num_lags_back):
    return FlowDataMaker(0, place_id, SPLIT_DATE, num_lags_back)


if __name__ == '__main__':
    flow_data_maker = default_bootstrap_flow_data_maker(EXAMPLE_PLACE_ID)
    x_train, y_train, x_test, y_test = flow_data_maker.get_train_and_test_inputs()
    print(x_train.head(), '\n')
    print(y_test.head(), '\n')
    print(flow_data_maker.individual_errors_without_interpolated_values([7] * len(y_test)).head())
