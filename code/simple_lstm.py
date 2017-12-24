from generic_nn import GenericNN

from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.models import Sequential


class SimpleLSTM(GenericNN):
    def build_model(self):
        model = Sequential()
        model.add(LSTM(self.lstm_state_size, input_shape=(self.num_lags, 1), return_sequences=False))
        model.add(Dense(units=self.num_outs, activation="linear"))
        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model
