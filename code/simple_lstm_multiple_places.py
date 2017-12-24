from keras.callbacks import ModelCheckpoint
import numpy as np
from generic_nn import GenericNN

from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.models import Sequential


class SimpleLSTMMultiplePlaces(GenericNN):
    def __init__(self, num_places, *args, **kwargs):
        GenericNN.__init__(self, *args, **kwargs)
        self.num_places = num_places

    def build_model(self):
        model = Sequential()
        model.add(LSTM(self.lstm_state_size, input_shape=(self.num_places, self.num_lags * 2),
                       return_sequences=False))
        model.add(Dense(units=self.num_places * 2, activation="linear"))
        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def prediction_stats(self, model):
        _, _, x_test, y_test = self.data_maker.get_train_and_test_inputs()
        predictions = model.predict(x_test)
        return self.data_maker.stats(predictions[:, 0::2], predictions[:, 1::2])

    def create_and_train_model(self):
        checkpoint_path = './hdf5/%d.hdf5' % self.unique_id
        print('Creating model %s' % checkpoint_path)
        x_train, y_train, _, _ = self.data_maker.get_train_and_test_inputs()
        model = self.build_model()
        history = model.fit(
            x_train,
            np.reshape(y_train, (y_train.shape[0], y_train.shape[1] * y_train.shape[2])),
            batch_size=self.mini_batch_size,
            epochs=self.num_epochs,
            validation_split=self.validation_split,
            callbacks=[ModelCheckpoint(
                checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')],
            verbose=2)
        model.load_weights(checkpoint_path)
        # self.plot_model_training(history)
        return model, history
