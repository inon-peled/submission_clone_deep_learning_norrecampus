import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (16, 10)

import tensorflow as tf
from sklearn.linear_model import LinearRegression
from datetime import timedelta
from keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.models import Sequential

# Prevent tensorflow from allocating the entire GPU memory at once
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

ONE_DAY_LAGS = 288
DATA_PATH = os.path.join('.', 'data', 'by_place_5min')  # Soft link to actual hp32:/mnt/sdc1/inon/norrecampus/data
EXAMPLE_PLACE_ID = 'ChIJZaR1M1hSUkYRxP0WkwYYy_k'
OUTPUT_DIR = os.path.join('.', 'output')

THRESHOLD_HIGH_OUTLIER = 110
LOSS = 'mse'
MINI_BATCH_SIZE = 512
NUM_EPOCHS = 100
VALIDATION_SPLIT = 0.2
NUM_LAGS = 12
LSTM_STATE_SIZE = NUM_LAGS
SPLIT_DATE = '2015-06-01'


class DefaultBootstrapper(object):
    def create_learner(self, place_id):
        return LearnerForFlowAndSpeedOneSegment(
            threshold_speed_outlier=THRESHOLD_HIGH_OUTLIER,
            loss=LOSS,
            place_id=place_id,
            mini_batch_size=MINI_BATCH_SIZE,
            num_epochs=NUM_EPOCHS,
            validation_split=VALIDATION_SPLIT,
            num_lags=NUM_LAGS,
            lstm_state_size=LSTM_STATE_SIZE,
            split_date=SPLIT_DATE)


def split_timestamp(ser):
    return pd.DataFrame(ser).assign(day_of_week=lambda df: df.index.dayofweek,
                                    time_of_day=lambda df: df.index.time).reset_index()


class LearnerForFlowAndSpeedOneSegment(object):
    def __init__(self,
                 threshold_speed_outlier,
                 loss,
                 place_id,
                 mini_batch_size,
                 num_epochs,
                 validation_split,
                 num_lags,
                 lstm_state_size,
                 split_date):
        self.model = None

        self.threshold_speed_outlier = threshold_speed_outlier
        self.loss = loss
        self.place_id = place_id
        self.mini_batch_size = mini_batch_size
        self.num_epochs = num_epochs
        self.validation_split = validation_split
        self.num_lags = num_lags
        self.lstm_state_size = lstm_state_size
        self.split_date = split_date
        self.checkpoint_basename = os.path.join(OUTPUT_DIR, 'simple_lstm_both_%s.best.hdf5' % self.place_id)

    def compute_error_statistics_and_plot_15jan2017(self, errors_df, plot_basename):
        errors_df.drop('error', axis=1)[ONE_DAY_LAGS * 14:ONE_DAY_LAGS * 15].plot()
        plt.legend()
        plt.savefig(os.path.join(OUTPUT_DIR, self.place_id + '_' + plot_basename))

        abs_errors = errors_df.error.abs()
        abs_errors_normalized = abs_errors / errors_df.speed_km_hr_true
        return {
            'corr': np.corrcoef(errors_df.speed_km_hr_predicted, errors_df.speed_km_hr_true)[0, 1],
            'mae': np.mean(abs_errors),
            'mape': np.mean(abs_errors_normalized),
            'mse': np.mean(abs_errors ** 2),
            'msne': np.mean(abs_errors_normalized ** 2),
            'rae': np.sum(abs_errors) / np.sum(
                np.abs(errors_df.speed_km_hr_true - np.mean(errors_df.speed_km_hr_true))),
            'rmse': np.sqrt(np.mean(abs_errors ** 2)),
            'rmsne': np.sqrt(np.mean(abs_errors_normalized ** 2)),
            'r2': max(0, 1 - np.sum(abs_errors ** 2) / np.sum(
                (errors_df.speed_km_hr_true - np.mean(errors_df.speed_km_hr_true)) ** 2))
        }

    def baseline_lr(self):
        X_train_normalized, Y_train_normalized, X_test_normalized, Y_test_normalized = self._get_train_and_test_inputs()
        trained_lr = LinearRegression(fit_intercept=False).fit(X_train_normalized.values, Y_train_normalized.values)
        lr_predictions_normalized = trained_lr.predict(X_test_normalized)
        errors_df = self._individual_errors_without_interpolated_values(lr_predictions_normalized)
        return self.compute_error_statistics_and_plot_15jan2017(errors_df, 'baseline_lr.png')

    def _get_detrend_factors_for_speed(self):
        ser_imp = self._speed_imputation_for_missing_values_and_outliers(self._get_speeds()).speed_km_hr
        df_train_period = ser_imp[lambda df: df.index < self.split_date]
        return df_train_period.groupby([df_train_period.index.dayofweek, df_train_period.index.time]).agg(
            ['mean', 'std']).reset_index().rename(columns={'level_0': 'day_of_week', 'level_1': 'time_of_day'})

    def detrend(self, ser):
        return pd.merge(left=split_timestamp(ser),
                        right=self._get_detrend_factors_for_speed(),
                        on=['day_of_week', 'time_of_day'],
                        how='inner') \
            .assign(speed_normalized=lambda df: (df.speed_km_hr - df['mean']) / df['std']) \
            .set_index('index') \
            .speed_normalized \
            .sort_index()

    def _map_back(self, detrended_series):
        return pd.merge(left=split_timestamp(detrended_series),
                        right=self._get_detrend_factors_for_speed(),
                        on=['day_of_week', 'time_of_day'],
                        how='inner') \
            .assign(speed_km_hr=lambda df: (df.speed_normalized * df['std']) + df['mean']) \
            .set_index('index') \
            .speed_km_hr \
            .sort_index()

    def _speed_imputation_for_missing_values_and_outliers(self, ser):
        missing_timestamps = pd.date_range(min(ser.index.date), max(ser.index.date) + timedelta(days=1),
                                           freq='5T').difference(ser.index)
        df_with_nans_where_missing = pd.DataFrame(ser).join(pd.DataFrame(index=missing_timestamps), how='outer').assign(
            original_value=lambda df: df.iloc[:, 0])
        return df_with_nans_where_missing.iloc[:, 0].mask(
            lambda df: df > self.threshold_speed_outlier).interpolate().to_frame().assign(
            original_value=df_with_nans_where_missing.original_value).assign(
            is_interpolated=lambda df: df.iloc[:, 0] != df.iloc[:, 1])

    def _flow_imputation_for_missing_values(self, ser):
        missing_timestamps = pd.date_range(min(ser.index.date), max(ser.index.date) + timedelta(days=1),
                                           freq='5T').difference(ser.index)
        df_with_nans_where_missing = pd.DataFrame(ser).join(pd.DataFrame(index=missing_timestamps), how='outer').assign(
            original_value=lambda df: df.iloc[:, 0])
        df_interpolated = df_with_nans_where_missing.iloc[:, 0].interpolate().round().astype(
            np.int64).to_frame().assign(original_value=df_with_nans_where_missing.original_value).assign(
            is_interpolated=lambda df: df.iloc[:, 0] != df.iloc[:, 1])
        assert all(df_interpolated.flow_decile.isin(range(10)))
        return df_interpolated

    def _get_flows(self):
        return pd.read_csv(
            os.path.join(DATA_PATH, self.place_id + '.csv'),
            parse_dates=['start_interval_s', 'end_interval_s'] \
            )[lambda df: df.start_interval_s >= '2015-01-01'] \
            .rename(columns={'start_interval_s': 't', 'flow_bucket': 'flow_decile'}) \
            .set_index('t') \
            .flow_decile

    def _get_speeds(self):
        return pd.read_csv(
            os.path.join(DATA_PATH, self.place_id + '.csv'),
            parse_dates=['start_interval_s', 'end_interval_s'] \
            )[lambda df: df.start_interval_s >= '2015-01-01'] \
            .rename(columns={'start_interval_s': 't', 'speed_mean': 'speed_km_hr'}) \
            .set_index('t') \
            .speed_km_hr

    def _get_train_and_test_inputs(self):
        def create_lags(ser, column_rename_letter):
            df = pd.concat([ser.shift(x) for x in range(self.num_lags + 1)], axis=1)[self.num_lags + 1:]
            df.columns = [column_rename_letter + str(i) for i in range(self.num_lags + 1)]
            return df

        speeds_interpolated_and_detrended = self.detrend(
            self._speed_imputation_for_missing_values_and_outliers(self._get_speeds()).speed_km_hr)
        flows_interpolated = self._flow_imputation_for_missing_values(self._get_flows()).flow_decile
        lags = create_lags(speeds_interpolated_and_detrended, 's').join(create_lags(flows_interpolated, 'f'))
        train = lags[lags.index < self.split_date]
        X_train = train.drop(['s0', 'f0'], axis=1)
        Y_train = train[['s0']]
        test = lags[lags.index >= self.split_date]
        X_test = test.drop(['s0', 'f0'], axis=1)
        Y_test = test[['s0']]
        ret = X_train, Y_train, X_test, Y_test
        assert not any(df.isnull().values.any() for df in ret)
        return ret

    def _build_model(self):
        self.model = Sequential()
        self.model.add(LSTM(self.lstm_state_size, input_shape=(self.num_lags * 2, 1), return_sequences=False))
        # model.add(Dropout(0.2))
        # model.add(Dense(units=1, kernel_regularizer=regularizers.l2(0.00001)))
        self.model.add(Dense(units=1,
                             activation="linear"))  # Linear activation, because speed RESIDUALS can have any sign.
        self.model.compile(loss=self.loss,
                           optimizer="rmsprop")  # TODO: try adam optimizer too, although rmsprop is the default go-to for RNN

    def load_best_model_from_disk(self):
        self._build_model()
        self.model.load_weights(self.checkpoint_basename)

    def create_and_train_model(self):
        self._build_model()
        X_train, Y_train, _, _ = self._get_train_and_test_inputs()
        self.model.fit(
            np.expand_dims(np.array(X_train.values), 2),
            Y_train.values,
            batch_size=self.mini_batch_size,
            epochs=self.num_epochs,
            validation_split=self.validation_split,
            # checkpoint best model
            callbacks=[ModelCheckpoint(
                self.checkpoint_basename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')],
            verbose=2)
        self.model.load_weights(self.checkpoint_basename)

    def predict(self):
        _, _, X_test_normalized, Y_test_normalized = self._get_train_and_test_inputs()
        predictions_normalized = self.model.predict(np.expand_dims(X_test_normalized, 2))
        errors_df = self._individual_errors_without_interpolated_values(predictions_normalized)
        return self.compute_error_statistics_and_plot_15jan2017(errors_df, 'deep_model.png')

    def _impute(self):
        raise NotImplementedError

    def _individual_errors_without_interpolated_values(self, predictions_normalized):
        _, _, _, Y_test_normalized = self._get_train_and_test_inputs()
        ser_Y_test = self._map_back(Y_test_normalized.iloc[:, 0].squeeze().rename('speed_normalized'))
        ser_predictions = self._map_back(pd.DataFrame(
            predictions_normalized, index=Y_test_normalized.index, columns=['speed_normalized']))
        interpolated_timestamps = self._speed_imputation_for_missing_values_and_outliers(self._get_speeds())[
            lambda df: df.is_interpolated].index
        return ser_Y_test.to_frame().join(ser_predictions.to_frame(), lsuffix='_true', rsuffix='_predicted').loc[
            lambda df: df.index.difference(interpolated_timestamps)].assign(
            error=lambda df: df.speed_km_hr_true - df.speed_km_hr_predicted)


if __name__ == '__main__':
    learner = DefaultBootstrapper().create_learner(EXAMPLE_PLACE_ID)
    learner.create_and_train_model()
    # learner.load_best_model_from_disk()
    print(learner.predict())

    print(learner.baseline_lr())
