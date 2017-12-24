from common import *

import random
from keras.callbacks import ModelCheckpoint
import numpy as np
from matplotlib import pyplot as plt


class GenericNN(object):
    def __init__(self, data_maker, lstm_state_size, num_lags, num_outs, loss, optimizer,
                 mini_batch_size, num_epochs, validation_split):
        self.mini_batch_size, self.num_epochs, self.validation_split, self.data_maker, \
            self.lstm_state_size, self.num_lags, self.num_outs, self.loss, self.optimizer = \
            mini_batch_size, num_epochs, validation_split, data_maker, lstm_state_size, num_lags, num_outs, \
            loss, optimizer
        self.unique_id = random.randint(2 ** 40, 2 ** 41)

    def build_model(self):
        raise NotImplementedError

    def plot_model_training(self, history):
        plt.plot(history.history['loss'], linestyle='--')
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss in Train Phase')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        # plt.show()
        fig_path = './figs/%d.png' % self.unique_id
        plt.savefig(fig_path)
        print('Saved', fig_path)

    def create_and_train_model(self):
        checkpoint_path = './hdf5/%d.hdf5' % self.unique_id
        print('Creating model %s' % checkpoint_path)
        x_train, y_train, _, _ = self.data_maker.get_train_and_test_inputs()
        model = self.build_model()
        history = model.fit(
            np.expand_dims(np.array(x_train.values), 2),
            y_train.values,
            batch_size=self.mini_batch_size,
            epochs=self.num_epochs,
            validation_split=self.validation_split,
            callbacks=[ModelCheckpoint(
                checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')],
            verbose=2)
        model.load_weights(checkpoint_path)
        # self.plot_model_training(history)
        return model, history

    def prediction_stats(self, model):
        _, _, x_test, y_test = self.data_maker.get_train_and_test_inputs()
        predictions = model.predict(np.expand_dims(x_test, 2)).flatten()
        return self.data_maker.stats(predictions)
