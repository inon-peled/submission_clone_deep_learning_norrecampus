import sys
import os

os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
PLACE_GROUP = sys.argv[2]

import pandas as pd
from place_groups import SELECTED_PLACES
from lstm_combined_dropout_regularization import LSTMSpeedsAndFlowsDropoutRegularization
from common import BEST_LSTM_STATE_SIZE, BEST_MINI_BATCH_SIZE, BEST_NUM_EPOCHS, BEST_NUM_PAST_LAGS

from matplotlib import pyplot as plt

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 16

import tensorflow as tf

# Prevent tensorflow from allocating the entire GPU memory at once
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

from speed_and_flow_data_maker import default_bootstrap_speed_and_flow_data_maker


def one_place(dropout_rate, l2_kernel_regularization_parameter, place_id, lstm_state_size, num_lags_back,
              mini_batch_size, num_epochs):
    nn = LSTMSpeedsAndFlowsDropoutRegularization(
        dropout_rate=dropout_rate,
        l2_kernel_regularization_parameter=l2_kernel_regularization_parameter,
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
    open('./res/%s_lstm_speeds_and_flows_%s_dropout_%f_reg_%f.txt' % (
        PLACE_GROUP, place_id, dropout_rate, l2_kernel_regularization_parameter), 'w').write(str(pred_stats))
    return pred_stats


def all_places(places, dropout_rate, l2_kernel_regularization_parameter, lstm_state_size, num_lags_back,
               mini_batch_size, num_epochs):
    print('------------------ all_places dropout %f reg %f ----------------' % (
        dropout_rate, l2_kernel_regularization_parameter))
    res = pd.concat([one_place(dropout_rate, l2_kernel_regularization_parameter, place_id, lstm_state_size,
                               num_lags_back, mini_batch_size, num_epochs)
                     for place_id in places]) \
        .mean()
    res.to_csv('%s_experiment_nn_lstm_speeds_and_flows_dropout_%f_reg_%f.csv' % (
        PLACE_GROUP, dropout_rate, l2_kernel_regularization_parameter))
    return res


if __name__ == '__main__':
    for drpout_rate in [0.1, 0.2, 0.4, 0.8, 0]:
        for l2_kernel_reg in [1E-2, 1E-4, 1E-6, 0]:
            # all_places(selected_middle_of_roads[:2], drpout_rate, l2_kernel_reg, 1, 1, 512, 1)
            all_places(SELECTED_PLACES[PLACE_GROUP], drpout_rate, l2_kernel_reg,
                       BEST_LSTM_STATE_SIZE, BEST_NUM_PAST_LAGS, BEST_MINI_BATCH_SIZE, BEST_NUM_EPOCHS)
