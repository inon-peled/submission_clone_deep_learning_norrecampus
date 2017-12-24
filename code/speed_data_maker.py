from common import *

import os
from datetime import timedelta
import pandas as pd


def _split_timestamp(ser):
    return pd.DataFrame(ser).assign(day_of_week=lambda df: df.index.dayofweek,
                                    time_of_day=lambda df: df.index.time).reset_index()


class SpeedDataMaker(object):
    def __init__(self, default_value_if_still_missing_after_interpolation,
                 place_id, split_date, threshold_high_outlier, num_lags_back):
        self.default_speed_if_still_missing_after_interpolation = default_value_if_still_missing_after_interpolation
        self.place_id = place_id
        self.split_date = split_date
        self.threshold_high_outlier = threshold_high_outlier
        self.num_lags_back = num_lags_back

    def _get_detrend_factors(self):
        ser_imp = self._imputation_for_missing_values_and_outliers(self._get_df()).speed_km_hr
        df_train_period = ser_imp[lambda df: df.index < self.split_date]
        return df_train_period\
            .groupby([df_train_period.index.dayofweek, df_train_period.index.time])\
            .agg(['mean', 'std'])\
            .reset_index()\
            .rename(columns={'level_0': 'day_of_week', 'level_1': 'time_of_day'})

    def detrend(self, ser):
        return pd.merge(left=_split_timestamp(ser),
                        right=self._get_detrend_factors(),
                        on=['day_of_week', 'time_of_day'],
                        how='inner') \
            .assign(speed_normalized=lambda df: (df.speed_km_hr - df['mean']) / df['std']) \
            .set_index('index') \
            .speed_normalized \
            .sort_index()

    def _map_back(self, detrended_series):
        return pd.merge(left=_split_timestamp(detrended_series),
                        right=self._get_detrend_factors(),
                        on=['day_of_week', 'time_of_day'],
                        how='inner') \
            .assign(speed_km_hr=lambda df: (df[detrended_series.name] * df['std']) + df['mean']) \
            .set_index('index') \
            .speed_km_hr \
            .sort_index()

    def _imputation_for_missing_values_and_outliers(self, ser):
        missing_timestamps = pd.date_range(min(ser.index.date), max(ser.index.date) + timedelta(days=1),
                                           freq='5T').difference(ser.index)
        df_with_nans_where_missing = pd.DataFrame(ser).join(pd.DataFrame(index=missing_timestamps), how='outer').assign(
            original_value=lambda df: df.iloc[:, 0])
        return df_with_nans_where_missing.iloc[:, 0] \
            .mask(lambda df: df > self.threshold_high_outlier) \
            .interpolate() \
            .fillna(self.default_speed_if_still_missing_after_interpolation) \
            .to_frame() \
            .assign(original_value=df_with_nans_where_missing.original_value) \
            .assign(is_interpolated=lambda df: df.iloc[:, 0] != df.iloc[:, 1])

    def _get_df(self):
        return pd.read_csv(
            os.path.join(DATA_PATH, self.place_id + '.csv'),
            parse_dates=['start_interval_s', 'end_interval_s']
        )[lambda df: df.start_interval_s >= '2015-01-01'] \
            .rename(columns={'start_interval_s': 't', 'speed_mean': 'speed_km_hr'}) \
            .set_index('t') \
            .speed_km_hr

    def get_train_and_test_inputs(self):
        speeds_interpolated_and_detrended = self.detrend(
            self._imputation_for_missing_values_and_outliers(self._get_df()).speed_km_hr)
        lags = pd.concat([speeds_interpolated_and_detrended.shift(x).rename('s%d'% x)
                          for x in range(self.num_lags_back + 1)], axis=1)[self.num_lags_back:]
        train = lags[lags.index < self.split_date]
        x_train = train.iloc[:, 1:]
        y_train = train.iloc[:, 0]
        test = lags[lags.index >= self.split_date]
        x_test = test.iloc[:, 1:]
        y_test = test.iloc[:, 0]
        return x_train, y_train, x_test, y_test

    def individual_errors_without_interpolated_values(self, predictions_normalized):
        _, _, _, y_test_normalized = self.get_train_and_test_inputs()
        ser_y_test = self._map_back(y_test_normalized)
        ser_predictions = self._map_back(pd.Series(
            predictions_normalized, index=y_test_normalized.index, name='speed_normalized'))
        interpolated_timestamps = self._imputation_for_missing_values_and_outliers(self._get_df())[
            lambda df: df.is_interpolated].index
        return ser_y_test.to_frame().join(ser_predictions.to_frame(), lsuffix='_true', rsuffix='_predicted').loc[
            lambda df: df.index.difference(interpolated_timestamps)].assign(
            error=lambda df: df.speed_km_hr_true - df.speed_km_hr_predicted)

    def stats(self, predictions):
        errors_df = self.individual_errors_without_interpolated_values(predictions)
        return compute_error_statistics(errors_df, 'speed_km_hr_true', 'speed_km_hr_predicted')


def default_bootstrap_speed_data_maker(place_id, num_lags_back):
    return SpeedDataMaker(0, place_id, SPLIT_DATE, THRESHOLD_SPEED_OUTLIER_KM_PER_HR, num_lags_back)


if __name__ == '__main__':
    speed_maker = default_bootstrap_speed_data_maker(EXAMPLE_PLACE_ID)
    x_train, y_train, x_test, y_test = speed_maker.get_train_and_test_inputs()
    print(x_train.head())
    print(y_test.head())
