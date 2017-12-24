import sys
PLACES_GROUP, GPU, HALF_SIZE = sys.argv[1], sys.argv[2], int(sys.argv[3])
# PLACES_GROUP, GPU, HALF_SIZE = 'junction', '1', 1

import os
os.environ['CUDA_VISIBLE_DEVICES'] = GPU

import pandas as pd
from place_groups import SELECTED_PLACES
from mixture_of_experts_nn import MixtureOfExpertsNN
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

from separate_speed_and_flow_data_maker import default_bootstrap_separate_speed_and_flow_data_maker


def one_place(half_lstm_state_size, place_id, lstm_state_size, num_lags_back, mini_batch_size, num_epochs):
    nn = MixtureOfExpertsNN(
        half_lstm_state_size=half_lstm_state_size,
        loss='mse',
        data_maker=default_bootstrap_separate_speed_and_flow_data_maker(place_id, num_lags_back),
        lstm_state_size=lstm_state_size,
        num_lags=num_lags_back,
        num_outs=2,
        optimizer='rmsprop',
        mini_batch_size=mini_batch_size,
        num_epochs=num_epochs,
        validation_split=0.2)
    model, history = nn.create_and_train_model()
    pred_stats = nn.prediction_stats(model)
    open('./res/mixture_of_experts_halfsize_%d_%s.txt' % (lstm_state_size, place_id), 'w').write(str(pred_stats))
    return pred_stats


def all_places(half_lstm_state_size, places_group, lstm_state_size, num_lags_back, mini_batch_size, num_epochs):
    res = pd.concat([one_place(half_lstm_state_size, place_id, lstm_state_size, num_lags_back, mini_batch_size, num_epochs)
                     for place_id in SELECTED_PLACES[places_group]]) \
        .mean()
    res.to_csv('%s_nn_mixture_of_experts_halfsize_%d.csv' % (places_group, half_lstm_state_size))
    return res


if __name__ == '__main__':
    # all_places(HALF_SIZE, PLACES_GROUP, 2, 1, 512, 2)
    all_places(HALF_SIZE, PLACES_GROUP, BEST_LSTM_STATE_SIZE, BEST_NUM_PAST_LAGS, BEST_MINI_BATCH_SIZE, BEST_NUM_EPOCHS)
