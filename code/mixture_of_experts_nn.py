from keras.callbacks import ModelCheckpoint
from generic_nn import GenericNN

import keras

from separate_speed_and_flow_data_maker import default_bootstrap_separate_speed_and_flow_data_maker
from common import *


class MixtureOfExpertsNN(GenericNN):
    def __init__(self, half_lstm_state_size, *args, **kwargs):
        GenericNN.__init__(self, *args, **kwargs)
        self.half_lstm_state_size = half_lstm_state_size

    def create_and_train_model(self):
        checkpoint_path = './hdf5/%d.hdf5' % self.unique_id
        print('Creating model %s' % checkpoint_path)
        x_train, y_train, _, _ = self.data_maker.get_train_and_test_inputs()
        model = self.build_model()
        history = model.fit(
            [np.swapaxes(np.expand_dims(np.array(e), 2), 1, 2) for e in x_train],
            np.swapaxes(np.array([np.array(e) for e in y_train]), 0, 1),
            batch_size=self.mini_batch_size,
            epochs=self.num_epochs,
            validation_split=self.validation_split,
            callbacks=[ModelCheckpoint(
                checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')],
            verbose=2)
        model.load_weights(checkpoint_path)
        # self.plot_model_training(history)
        return model, history

    def build_model(self):
        input_speed = keras.layers.Input(shape=(None, self.num_lags))
        input_flow = keras.layers.Input(shape=(None, self.num_lags))
        state_size = self.lstm_state_size // 2 if self.half_lstm_state_size else self.lstm_state_size
        lstm_speed = keras.layers.recurrent.LSTM(state_size, input_shape=(self.num_lags, 1), return_sequences=False)(
            input_speed)
        lstm_flow = keras.layers.recurrent.LSTM(state_size, input_shape=(self.num_lags, 1), return_sequences=False)(
            input_flow)
        out = keras.layers.Dense(units=self.num_outs, activation="linear")(keras.layers.add([lstm_speed, lstm_flow]))
        model = keras.Model(inputs=[input_speed, input_flow], output=out)
        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def prediction_stats(self, model):
        _, _, x_test, y_test = self.data_maker.get_train_and_test_inputs()
        predictions = model.predict([np.swapaxes(np.expand_dims(np.array(e), 2), 1, 2) for e in x_test])
        return self.data_maker.stats(predictions[:, 0], predictions[:, 1])


if __name__ == "__main__":
    MixtureOfExpertsNN(
        half_lstm_state_size=True,
        data_maker=default_bootstrap_separate_speed_and_flow_data_maker(EXAMPLE_PLACE_ID, BEST_NUM_PAST_LAGS),
        lstm_state_size=BEST_LSTM_STATE_SIZE,
        num_lags=BEST_NUM_PAST_LAGS,
        num_outs=2,
        loss="mse",
        optimizer="rmsprop",
        mini_batch_size=BEST_MINI_BATCH_SIZE,
        num_epochs=BEST_NUM_EPOCHS,
        validation_split=0.2
    ).build_model()
