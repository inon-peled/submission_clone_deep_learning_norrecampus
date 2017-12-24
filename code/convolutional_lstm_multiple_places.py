from keras.callbacks import ModelCheckpoint
import numpy as np
from generic_nn import GenericNN

from keras.layers.core import Dense
from keras.layers import ConvLSTM2D
from keras.models import Sequential


class ConvolutionalLSTMMultiplePlaces(GenericNN):
    def __init__(self, conv_filters, num_places, *args, **kwargs):
        GenericNN.__init__(self, *args, **kwargs)
        self.conv_filters = conv_filters
        self.num_places = num_places

    def build_model(self):
        model = Sequential()
        model.add(ConvLSTM2D(
            filters=self.conv_filters,
            kernel_size=(5, 1),
            # All places are arranged one dimensionally as a vector of length num_places,
            # thus the two dimensional structure of the map is ignored for now.
            input_shape=(self.num_lags * 2, self.num_places, 1, 1),
            padding='same',
            return_sequences=False))
        model.add(Dense(units=2, activation="linear", input_shape=(None, self.num_places * 2)))
        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def prediction_stats(self, model):
        _, _, x_test, y_test = self.data_maker.get_train_and_test_inputs()
        predictions = model.predict(np.expand_dims(np.expand_dims(np.swapaxes(x_test, 1, 2), 3), 4))
        return self.data_maker.stats(np.squeeze(predictions[:, :, :, 0::2]),
                                     np.squeeze(predictions[:, :, :, 1::2]))

    def create_and_train_model(self):
        checkpoint_path = './hdf5/%d.hdf5' % self.unique_id
        print('Creating model %s' % checkpoint_path)
        x_train, y_train, _, _ = self.data_maker.get_train_and_test_inputs()
        model = self.build_model()
        model.summary()
        history = model.fit(
            np.expand_dims(np.expand_dims(np.swapaxes(x_train, 1, 2), 3), 4),
            np.expand_dims(y_train, 2),
            batch_size=self.mini_batch_size,
            epochs=self.num_epochs,
            validation_split=self.validation_split,
            callbacks=[ModelCheckpoint(
                checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')],
            verbose=2)
        model.load_weights(checkpoint_path)
        # self.plot_model_training(history)
        return model, history
