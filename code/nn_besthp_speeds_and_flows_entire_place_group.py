import sys
PLACES_GROUP_NAME, GPU = sys.argv[1:]
# PLACES_GROUP_NAME, GPU = 'junction', '1'

import os
os.environ['CUDA_VISIBLE_DEVICES'] = GPU

from place_groups import SELECTED_PLACES
from simple_lstm_multiple_places import SimpleLSTMMultiplePlaces
from matplotlib import pyplot as plt

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 16

# Prevent tensorflow from allocating the entire GPU memory at once
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

from common import BEST_NUM_PAST_LAGS, BEST_LSTM_STATE_SIZE, BEST_MINI_BATCH_SIZE, BEST_NUM_EPOCHS

from multiple_places_at_once_speed_and_flow_data_maker import \
    default_bootstrap_multiple_places_at_once_speed_and_flow_data_maker


def process(place_group_name, places_group, lstm_state_size, num_lags_back, mini_batch_size, num_epochs):
    nn = SimpleLSTMMultiplePlaces(
        num_places=len(places_group),
        loss='mse',
        data_maker=default_bootstrap_multiple_places_at_once_speed_and_flow_data_maker(places_group, num_lags_back),
        lstm_state_size=lstm_state_size,
        num_lags=num_lags_back,
        num_outs=len(places_group) * 2,
        optimizer='rmsprop',
        mini_batch_size=mini_batch_size,
        num_epochs=num_epochs,
        validation_split=0.2)
    model, history = nn.create_and_train_model()
    individual_pred_stats = nn.prediction_stats(model)
    open(('./res/individual_%s_simple_lstm_multiple_places_at_once.txt' % (place_group_name,)), 'w')\
        .write(str(individual_pred_stats))
    open(('./%s_simple_lstm_multiple_places_at_once.txt' % (place_group_name,)), 'w')\
        .write(str(individual_pred_stats.mean()))


if __name__ == '__main__':
    plcs_grp = SELECTED_PLACES[PLACES_GROUP_NAME]
    # process(PLACES_GROUP_NAME, plcs_grp[:3], 1, 2, 512, 1)
    process(PLACES_GROUP_NAME, plcs_grp,
            BEST_LSTM_STATE_SIZE * len(plcs_grp), BEST_NUM_PAST_LAGS, BEST_MINI_BATCH_SIZE, int(BEST_NUM_EPOCHS * 1.5))
