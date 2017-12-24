import sys
import os

os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[2]

import pandas as pd
from place_groups import selected_junctions, selected_middle_of_roads
from simple_lstm import SimpleLSTM

from matplotlib import pyplot as plt

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 16

# Prevent tensorflow from allocating the entire GPU memory at once
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

from speed_data_maker import default_bootstrap_speed_data_maker


def one_place(place_id, lstm_state_size, num_lags_back, mini_batch_size, num_epochs):
    nn = SimpleLSTM(
        loss='mse',
        data_maker=default_bootstrap_speed_data_maker(place_id, num_lags_back + 1),
        lstm_state_size=lstm_state_size,
        num_lags=num_lags_back + 1,
        num_outs=1,
        optimizer='rmsprop',
        mini_batch_size=mini_batch_size,
        num_epochs=num_epochs,
        validation_split=0.2)
    model, history = nn.create_and_train_model()
    return nn.prediction_stats(model)


def all_places(places, lstm_state_size, num_lags_back, mini_batch_size, num_epochs):
    results = pd.concat([
        pd.DataFrame(
            one_place(place_id, lstm_state_size, num_lags_back, mini_batch_size, num_epochs),
            index=[place_id])
        for place_id in places]) \
        .mean()
    return results


def explore_parameters(places, mini_batch_size):
    res_fname = 'res_minibatchsize_%s.csv' % mini_batch_size
    open(res_fname, 'w').write('lstm_state_size,num_lags_back,mini_batch_size,num_epochs,corr,mae,r2,rmse\n')
    for lstm_state_size in [10, 20, 30]:
        for num_lags_back in [3, 6, 12]:
            for num_epochs in [50, 100, 150]:
                res = all_places(places, lstm_state_size, num_lags_back, mini_batch_size, num_epochs)
                open(res_fname, 'a').write(','.join(str(s) for s in [
                    lstm_state_size, num_lags_back, mini_batch_size, num_epochs, res['corr'], res['mae'], res['r2'],
                    res['rmse']
                ]) + '\n')


if __name__ == '__main__':
    explore_parameters(selected_middle_of_roads[:3] + selected_junctions[:3], mini_batch_size=int(sys.argv[1]))
