import numpy as np
from generic_nn import GenericNN

from keras.regularizers import l2
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential


class LSTMSpeedsAndFlowsDropoutRegularization(GenericNN):
    def __init__(self, dropout_rate, l2_kernel_regularization_parameter, *args, **kwargs):
        GenericNN.__init__(self, *args, **kwargs)
        self.dropout_rate = dropout_rate
        self.l2_kernel_regularization_parameter = l2_kernel_regularization_parameter

    def build_model(self):
        model = Sequential()
        model.add(LSTM(self.lstm_state_size, input_shape=(self.num_lags * 2, 1), return_sequences=False,
                       dropout=self.dropout_rate))
        model.add(Dense(units=self.num_outs, activation='linear', kernel_regularizer=l2(
            self.l2_kernel_regularization_parameter) if self.l2_kernel_regularization_parameter else None))
        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def prediction_stats(self, model):
        _, _, x_test, y_test = self.data_maker.get_train_and_test_inputs()
        predictions = model.predict(np.expand_dims(x_test, 2))
        return self.data_maker.stats(predictions[:, 0], predictions[:, 1])
