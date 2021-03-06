import sys
PLACES_GROUP, GPU = sys.argv[1:]
# PLACES_GROUP, GPU = 'junction', '1'

import os
os.environ['CUDA_VISIBLE_DEVICES'] = GPU

import pandas as pd
from place_groups import SELECTED_PLACES
from simple_lstm_speeds_and_flows import SimpleLSTMSpeedsAndFlows
from common import BEST_LSTM_STATE_SIZE, BEST_MINI_BATCH_SIZE, BEST_NUM_EPOCHS, BEST_NUM_PAST_LAGS

from matplotlib import pyplot as plt

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 16

# Prevent tensorflow from allocating the entire GPU memory at once
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

from speed_and_flow_data_maker import default_bootstrap_speed_and_flow_data_maker

def one_place(place_id, lstm_state_size, num_lags_back, mini_batch_size, num_epochs):
    nn = SimpleLSTMSpeedsAndFlows(
        loss='mse',
        data_maker=default_bootstrap_speed_and_flow_data_maker(place_id, num_lags_back),
        lstm_state_size=lstm_state_size,
        num_lags=num_lags_back,
        num_outs=2,
        optimizer='rmsprop',
        mini_batch_size=mini_batch_size,
        num_epochs=num_epochs,
        validation_split=0.2)
    model, history = nn.create_and_train_model()
    pred_stats = nn.prediction_stats(model)
    open('./res/simple_lstm_speeds_and_flows_%s.txt' % place_id, 'w').write(str(pred_stats))
    return pred_stats


def all_places(places_group, lstm_state_size, num_lags_back, mini_batch_size, num_epochs):
    res = pd.concat([one_place(place_id, lstm_state_size, num_lags_back, mini_batch_size, num_epochs)
                     for place_id in SELECTED_PLACES[places_group]]) \
        .mean()
    res.to_csv('%s_nn_simple_lstm_speeds_and_flows.csv' % places_group)
    return res


if __name__ == '__main__':
    # all_places(PLACES_GROUP, 1, 1, 512, 1)
    all_places(PLACES_GROUP, BEST_LSTM_STATE_SIZE, BEST_NUM_PAST_LAGS, BEST_MINI_BATCH_SIZE, BEST_NUM_EPOCHS)
