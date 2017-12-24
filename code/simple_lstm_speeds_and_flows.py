import numpy as np
from generic_nn import GenericNN

from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.models import Sequential


class SimpleLSTMSpeedsAndFlows(GenericNN):
    def build_model(self):
        model = Sequential()
        model.add(LSTM(self.lstm_state_size, input_shape=(self.num_lags * 2, 1), return_sequences=False))
        model.add(Dense(units=self.num_outs, activation="linear"))
        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def prediction_stats(self, model):
        _, _, x_test, y_test = self.data_maker.get_train_and_test_inputs()
        predictions = model.predict(np.expand_dims(x_test, 2))
        return self.data_maker.stats(predictions[:, 0], predictions[:, 1])
